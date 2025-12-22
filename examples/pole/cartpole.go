package pole

import (
	"context"
	"fmt"
	"github.com/yaricom/goNEAT/v4/experiment"
	"github.com/yaricom/goNEAT/v4/experiment/utils"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
)

type cartPoleGenerationEvaluator struct {
	// The output path to store execution results
	OutputPath string

	// The flag to indicate if cart emulator should be started from random position
	RandomStart bool
	// The number of emulation steps to be done balancing pole to win
	WinBalancingSteps int

	// The maximal number of parallel workers for fitness evaluation
	MaxWorkers int
}

// NewCartPoleGenerationEvaluator is to create generations evaluator for single-pole balancing experiment.
// This experiment performs evolution on single pole balancing task in order to produce appropriate genome.
func NewCartPoleGenerationEvaluator(outDir string, randomStart bool, winBalanceSteps int) experiment.GenerationEvaluator {
	return &cartPoleGenerationEvaluator{
		OutputPath:        outDir,
		RandomStart:       randomStart,
		WinBalancingSteps: winBalanceSteps,
		MaxWorkers:        1,
	}
}

// GenerationEvaluate evaluates one epoch for given population and prints results into output directory if any.
func (e *cartPoleGenerationEvaluator) GenerationEvaluate(ctx context.Context, pop *genetics.Population, epoch *experiment.Generation) error {
	options, ok := neat.FromContext(ctx)
	if !ok {
		return neat.ErrNEATOptionsNotFound
	}
	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		res, err := OrganismEvaluate(org, e.WinBalancingSteps, e.RandomStart)
		if err != nil {
			return err
		}

		if res && (epoch.Champion == nil || org.Fitness > epoch.Champion.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = options.PopSize*epoch.Id + org.Genotype.Id
			epoch.Champion = org
			if epoch.WinnerNodes == 7 {
				// You could dump out optimal genomes here if desired
				if optPath, err := utils.WriteGenomePlain("pole1_optimal", e.OutputPath, org, epoch); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
				} else {
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", optPath))
				}
			}
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
