package pole

import (
	"context"
	"fmt"
	"github.com/yaricom/goNEAT/v4/experiment"
	"github.com/yaricom/goNEAT/v4/experiment/utils"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"sync"
)

type cartPoleParallelGenerationEvaluator struct {
	cartPoleGenerationEvaluator
}

type evaluationJobResult struct {
	genomeId int
	fitness  float64
	error    float64
	winner   bool
	err      error
}

// NewCartPoleParallelGenerationEvaluator is to create generations evaluator for single-pole balancing experiment.
// This experiment performs evolution on single pole balancing task in order to produce appropriate genome.
func NewCartPoleParallelGenerationEvaluator(outDir string, randomStart bool, winBalanceSteps int, maxWorkers int) experiment.GenerationEvaluator {
	return &cartPoleParallelGenerationEvaluator{
		cartPoleGenerationEvaluator{
			OutputPath:        outDir,
			RandomStart:       randomStart,
			WinBalancingSteps: winBalanceSteps,
			MaxWorkers:        maxWorkers,
		},
	}
}

type evaluationJob struct {
	organism *genetics.Organism
}

func worker(winnerBalancingSteps int, randomStart bool, jobs <-chan evaluationJob, resChan chan<- evaluationJobResult, wg *sync.WaitGroup) {
	defer wg.Done()
	// execute evaluation jobs
	for job := range jobs {
		// create simulator and evaluate
		winner, err := OrganismEvaluate(job.organism, winnerBalancingSteps, randomStart)
		if err != nil {
			resChan <- evaluationJobResult{err: err}
			return
		}

		// create result
		result := evaluationJobResult{
			genomeId: job.organism.Genotype.Id,
			fitness:  job.organism.Fitness,
			error:    job.organism.Error,
			winner:   winner,
		}
		resChan <- result
	}
}

// GenerationEvaluate evaluates one epoch for given population and prints results into output directory if any.
func (e *cartPoleParallelGenerationEvaluator) GenerationEvaluate(ctx context.Context, pop *genetics.Population, epoch *experiment.Generation) error {
	options, ok := neat.FromContext(ctx)
	if !ok {
		return neat.ErrNEATOptionsNotFound
	}

	organismMapping := make(map[int]*genetics.Organism)

	popSize := len(pop.Organisms)
	resChan := make(chan evaluationJobResult, popSize)
	jobsChan := make(chan evaluationJob, popSize)
	// The wait group to wait for all GO routines
	var wg sync.WaitGroup

	// Create pool of workers
	for i := 0; i < e.MaxWorkers; i++ {
		wg.Add(1)
		go worker(e.WinBalancingSteps, e.RandomStart, jobsChan, resChan, &wg)
	}

	// Evaluate each organism in generation
	for _, org := range pop.Organisms {
		if _, ok = organismMapping[org.Genotype.Id]; ok {
			return fmt.Errorf("organism with %d already exists in mapping", org.Genotype.Id)
		}
		organismMapping[org.Genotype.Id] = org
		// create and publish fitness evaluation job
		jobsChan <- evaluationJob{
			organism: org,
		}
	}
	// close jobs channel
	close(jobsChan)

	// wait for evaluation results
	wg.Wait()
	close(resChan)

	for result := range resChan {
		if result.err != nil {
			return result.err
		}
		// find and update original organism
		org, ok := organismMapping[result.genomeId]
		if ok {
			org.Fitness = result.fitness
			org.Error = result.error
		} else {
			return fmt.Errorf("organism not found in mapping for id: %d", result.genomeId)
		}

		if result.winner && (epoch.Champion == nil || org.Fitness > epoch.Champion.Fitness) {
			// This will be winner in Markov case
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = options.PopSize*epoch.Id + org.Genotype.Id
			epoch.Champion = org
			org.IsWinner = true
		}
	}

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(pop)

	// Only print to file every print_every generation
	if epoch.Solved || epoch.Id%options.PrintEvery == 0 {
		if _, err := utils.WritePopulationPlain(e.OutputPath, pop, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
			return err
		}
	}

	if epoch.Solved {
		// print winner organism's statistics
		org := epoch.Champion
		utils.PrintActivationDepth(org, true)

		genomeFile := "pole1_winner_genome"
		// Prints the winner organism to file!
		if orgPath, err := utils.WriteGenomePlain(genomeFile, e.OutputPath, org, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's genome, reason: %s\n", err))
		} else {
			neat.InfoLog(fmt.Sprintf("Generation #%d winner's genome dumped to: %s\n", epoch.Id, orgPath))
		}

		// Prints the winner organism's phenotype to the Cytoscape JSON file!
		if orgPath, err := utils.WriteGenomeCytoscapeJSON(genomeFile, e.OutputPath, org, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's phenome Cytoscape JSON graph, reason: %s\n", err))
		} else {
			neat.InfoLog(fmt.Sprintf("Generation #%d winner's phenome Cytoscape JSON graph dumped to: %s\n",
				epoch.Id, orgPath))
		}
	}

	return nil
}
