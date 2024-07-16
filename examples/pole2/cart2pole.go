package pole2

import (
	"context"
	"fmt"
	"github.com/yaricom/goNEAT/v4/experiment"
	"github.com/yaricom/goNEAT/v4/experiment/utils"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
)

type cartDoublePoleGenerationEvaluator struct {
	// The output path to store execution results
	OutputPath string
	// The flag to indicate whether to apply Markov evaluation variant
	Markov bool

	// The flag to indicate whether to use continuous activation or discrete
	ActionType ActionType
}

// NewCartDoublePoleGenerationEvaluator is the generations evaluator for double-pole balancing experiment: both Markov and non-Markov versions
func NewCartDoublePoleGenerationEvaluator(outDir string, markov bool, actionType ActionType) experiment.GenerationEvaluator {
	return &cartDoublePoleGenerationEvaluator{
		OutputPath: outDir,
		Markov:     markov,
		ActionType: actionType,
	}
}

// GenerationEvaluate Perform evaluation of one epoch on double pole balancing
func (e *cartDoublePoleGenerationEvaluator) GenerationEvaluate(ctx context.Context, pop *genetics.Population, epoch *experiment.Generation) error {
	options, ok := neat.FromContext(ctx)
	if !ok {
		return neat.ErrNEATOptionsNotFound
	}
	cartPole := NewCartPole(e.Markov)

	cartPole.nonMarkovLong = false
	cartPole.generalizationTest = false

	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		winner, err := OrganismEvaluate(org, cartPole, e.ActionType)
		if err != nil {
			return err
		}

		if winner && (epoch.Champion == nil || org.Fitness > epoch.Champion.Fitness) {
			// This will be winner in Markov case
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = options.PopSize*epoch.Id + org.Genotype.Id
			epoch.Champion = org
			org.IsWinner = true
		}
	}

	// Check for winner in Non-Markov case
	if !e.Markov {
		epoch.Solved = false
		// evaluate generalization tests
		if champion, err := EvaluateOrganismGeneralization(pop.Species, cartPole, e.ActionType); err != nil {
			return err
		} else if champion.IsWinner {
			epoch.Solved = true
			epoch.WinnerNodes = len(champion.Genotype.Nodes)
			epoch.WinnerGenes = champion.Genotype.Extrons()
			epoch.WinnerEvals = options.PopSize*epoch.Id + champion.Genotype.Id
			epoch.Champion = champion
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

		genomeFile := "pole2_winner_genome"
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
