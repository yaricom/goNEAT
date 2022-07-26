// Package xor defines the XOR experiment which serves to actually check that network topology actually evolves and
// everything works as expected.
// Because XOR is not linearly separable, a neural network requires hidden units to solve it. The two inputs must be
// combined at some hidden unit, as opposed to only at the output node, because there is no function over a linear
// combination of the inputs that can separate the inputs into the proper classes. These structural requirements make
// XOR suitable for testing NEAT’s ability to evolve structure.
package xor

import (
	"context"
	"fmt"
	"github.com/yaricom/goNEAT/v3/experiment"
	"github.com/yaricom/goNEAT/v3/experiment/utils"
	"github.com/yaricom/goNEAT/v3/neat"
	"github.com/yaricom/goNEAT/v3/neat/genetics"
	"math"
)

// The fitness threshold value for successful solver
const fitnessThreshold = 15.5

type xorGenerationEvaluator struct {
	// The output path to store execution results
	OutputPath string
}

// NewXORGenerationEvaluator is to create new generations' evaluator to be used for the XOR experiment execution.
// XOR is very simple and does not make a very interesting scientific experiment; however, it is a good way to
// check whether your system works.
// Make sure recurrency is disabled for the XOR test. If NEAT is able to add recurrent connections, it may solve XOR by
// memorizing the order of the training set. (Which is why you may even want to randomize order to be most safe) All
// documented experiments with XOR are without recurrent connections. Interestingly, XOR can be solved by a recurrent
// network with no hidden nodes.
//
// This method performs evolution on XOR for specified number of generations and output results into outDirPath
// It also returns number of nodes, genes, and evaluations performed per each run (context.NumRuns)
func NewXORGenerationEvaluator(outputPath string) experiment.GenerationEvaluator {
	return &xorGenerationEvaluator{OutputPath: outputPath}
}

// GenerationEvaluate This method evaluates one epoch for given population and prints results into output directory if any.
func (e *xorGenerationEvaluator) GenerationEvaluate(ctx context.Context, pop *genetics.Population, epoch *experiment.Generation) error {
	options, ok := neat.FromContext(ctx)
	if !ok {
		return neat.ErrNEATOptionsNotFound
	}
	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		res, err := e.orgEvaluate(org)
		if err != nil {
			return err
		}

		if res && (epoch.Champion == nil || org.Fitness > epoch.Champion.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = options.PopSize*epoch.Id + org.Genotype.Id
			epoch.Champion = org
			if epoch.WinnerNodes == 5 {
				// You could dump out optimal genomes here if desired
				if optPath, err := utils.WriteGenomePlain("xor_optimal", e.OutputPath, org, epoch); err != nil {
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
		// print winner organism
		org := epoch.Champion
		utils.PrintActivationDepth(org, true)

		genomeFile := "xor_winner_genome"
		// Prints the winner organism's Genome to the file!
		if orgPath, err := utils.WriteGenomePlain(genomeFile, e.OutputPath, org, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's genome, reason: %s\n", err))
		} else {
			neat.InfoLog(fmt.Sprintf("Generation #%d winner's genome dumped to: %s\n", epoch.Id, orgPath))
		}

		// Prints the winner organism's phenotype to the DOT file!
		if orgPath, err := utils.WriteGenomeDOT(genomeFile, e.OutputPath, org, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's phenome DOT graph, reason: %s\n", err))
		} else {
			neat.InfoLog(fmt.Sprintf("Generation #%d winner's phenome DOT graph dumped to: %s\n",
				epoch.Id, orgPath))
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

// orgEvaluate evaluates fitness of the provided organism
func (e *xorGenerationEvaluator) orgEvaluate(organism *genetics.Organism) (bool, error) {
	// The four possible input combinations to xor
	// The first number is for biasing
	in := [][]float64{
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0},
		{1.0, 1.0, 1.0}}

	phenotype, err := organism.Phenotype()
	if err != nil {
		return false, err
	}

	netDepth, err := phenotype.MaxActivationDepthWithCap(0) // The max depth of the network to be activated
	if err != nil {
		neat.WarnLog(fmt.Sprintf(
			"Failed to estimate maximal depth of the network with loop:\n%s\nUsing default depth: %d",
			organism.Genotype, netDepth))
	}
	neat.DebugLog(fmt.Sprintf("Network depth: %d for organism: %d\n", netDepth, organism.Genotype.Id))
	if netDepth == 0 {
		neat.DebugLog(fmt.Sprintf("ALERT: Network depth is ZERO for Genome: %s", organism.Genotype))
		return false, nil
	}

	success := false          // Check for successful activation
	out := make([]float64, 4) // The four outputs

	// Load and activate the network on each input
	for count := 0; count < 4; count++ {
		if err = phenotype.LoadSensors(in[count]); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to load sensors: %s", err))
			return false, err
		}

		// Use depth to ensure full relaxation
		if success, err = phenotype.ForwardSteps(netDepth); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to activate network: %s", err))
			return false, err
		}
		out[count] = phenotype.Outputs[0].Activation

		// Flush network for subsequent use
		if _, err = phenotype.Flush(); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to flush network: %s", err))
			return false, err
		}
	}

	if success {
		// Mean Squared Error
		errorSum := math.Abs(out[0]) + math.Abs(1.0-out[1]) + math.Abs(1.0-out[2]) + math.Abs(out[3]) // ideal == 0
		target := 4.0 - errorSum                                                                      // ideal == 4.0
		organism.Fitness = math.Pow(4.0-errorSum, 2.0)
		organism.Error = math.Pow(4.0-target, 2.0)
	} else {
		// The network is flawed (shouldn't happen) - flag as anomaly
		organism.Error = 1.0
		organism.Fitness = 0.0
	}

	if organism.Fitness > fitnessThreshold {
		organism.IsWinner = true
		neat.InfoLog(fmt.Sprintf(">>>> Output activations: %e\n", out))

	} else {
		organism.IsWinner = false
	}
	return organism.IsWinner, nil
}
