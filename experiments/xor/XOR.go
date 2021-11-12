// Package xor defines the XOR experiment which serves to actually check that network topology actually evolves and
// everything works as expected.
// Because XOR is not linearly separable, a neural network requires hidden units to solve it. The two inputs must be
// combined at some hidden unit, as opposed to only at the output node, because there is no function over a linear
// combination of the inputs that can separate the inputs into the proper classes. These structural requirements make
// XOR suitable for testing NEAT’s ability to evolve structure.
package xor

import (
	"fmt"
	"github.com/yaricom/goNEAT/v2/experiment"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"math"
	"os"
)

// The fitness threshold value for successful solver
const fitnessThreshold = 15.5

type xorGenerationEvaluator struct {
	// The output path to store execution results
	OutputPath string
}

// NewXORGenerationEvaluator is to create new generations evaluator to be used for the XOR experiment execution.
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
func (e *xorGenerationEvaluator) GenerationEvaluate(pop *genetics.Population, epoch *experiment.Generation, context *neat.Options) (err error) {
	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		res, err := e.orgEvaluate(org)
		if err != nil {
			return err
		}

		if res && (epoch.Best == nil || org.Fitness > epoch.Best.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize*epoch.Id + org.Genotype.Id
			epoch.Best = org
			if epoch.WinnerNodes == 5 {
				// You could dump out optimal genomes here if desired
				optPath := fmt.Sprintf("%s/%s_%d-%d", experiment.OutDirForTrial(e.OutputPath, epoch.TrialId),
					"xor_optimal", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				if file, err := os.Create(optPath); err != nil {
					return err
				} else if err = org.Genotype.Write(file); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
					return err
				} else {
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", optPath))
				}
			}
		}
	}

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(pop)

	// Only print to file every print_every generations
	if epoch.Solved || epoch.Id%context.PrintEvery == 0 {
		popPath := fmt.Sprintf("%s/gen_%d", experiment.OutDirForTrial(e.OutputPath, epoch.TrialId), epoch.Id)
		if file, err := os.Create(popPath); err != nil {
			return err
		} else if err = pop.WriteBySpecies(file); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
			return err
		}
	}

	if epoch.Solved {
		// print winner organism
		for _, org := range pop.Organisms {
			if org.IsWinner {
				// Prints the winner organism's Genome to the file!
				orgPath := fmt.Sprintf("%s/%s_%d-%d", experiment.OutDirForTrial(e.OutputPath, epoch.TrialId),
					"xor_winner_genome", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				if file, err := os.Create(orgPath); err != nil {
					return err
				} else if err = org.Genotype.Write(file); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's genome, reason: %s\n", err))
					return err
				} else {
					neat.InfoLog(fmt.Sprintf("Generation #%d winner's genome dumped to: %s\n", epoch.Id, orgPath))
				}
				// Prints the winner organism's Phenotype to the DOT file!
				orgPath = fmt.Sprintf("%s/%s_%d-%d.dot", experiment.OutDirForTrial(e.OutputPath, epoch.TrialId),
					"xor_winner_phenome", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				if file, err := os.Create(orgPath); err != nil {
					return err
				} else if err = org.Phenotype.WriteDOT(file); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's phenome, reason: %s\n", err))
					return err
				} else {
					neat.InfoLog(fmt.Sprintf("Generation #%d winner's phenome DOT graph dumped to: %s\n",
						epoch.Id, orgPath))
				}

				// Prints the winner organism's Phenotype to the Cytoscape JSON file!
				orgPath = fmt.Sprintf("%s/%s_%d-%d.cyjs", experiment.OutDirForTrial(e.OutputPath, epoch.TrialId),
					"xor_winner_phenome", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				if file, err := os.Create(orgPath); err != nil {
					return err
				} else if err = org.Phenotype.WriteCytoscapeJSON(file); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's phenome, reason: %s\n", err))
					return err
				} else {
					neat.InfoLog(fmt.Sprintf("Generation #%d winner's phenome Cytoscape JSON graph dumped to: %s\n",
						epoch.Id, orgPath))
				}

				break
			}
		}
	}

	return err
}

// This methods evaluates provided organism
func (e *xorGenerationEvaluator) orgEvaluate(organism *genetics.Organism) (bool, error) {
	// The four possible input combinations to xor
	// The first number is for biasing
	in := [][]float64{
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0},
		{1.0, 1.0, 1.0}}

	netDepth, err := organism.Phenotype.MaxActivationDepthFast() // The max depth of the network to be activated
	if err != nil {
		neat.WarnLog(
			fmt.Sprintf("Failed to estimate maximal depth of the network with loop:\n%s\nUsing default dpeth: %d",
				organism.Genotype, netDepth))
	}
	neat.DebugLog(fmt.Sprintf("Network depth: %d for organism: %d\n", netDepth, organism.Genotype.Id))
	if netDepth == 0 {
		neat.DebugLog(fmt.Sprintf("ALERT: Network depth is ZERO for Genome: %s", organism.Genotype))
	}

	success := false          // Check for successful activation
	out := make([]float64, 4) // The four outputs

	// Load and activate the network on each input
	for count := 0; count < 4; count++ {
		if err := organism.Phenotype.LoadSensors(in[count]); err != nil {
			neat.ErrorLog("Failed to load sensors")
			return false, err
		}

		// Relax net and get output
		success, err = organism.Phenotype.Activate()
		if err != nil {
			neat.ErrorLog("Failed to activate network")
			return false, err
		}

		// use depth to ensure relaxation
		for relax := 0; relax <= netDepth; relax++ {
			success, err = organism.Phenotype.Activate()
			if err != nil {
				neat.ErrorLog("Failed to activate network")
				return false, err
			}
		}
		out[count] = organism.Phenotype.Outputs[0].Activation

		if _, err := organism.Phenotype.Flush(); err != nil {
			neat.ErrorLog("Failed to flush network")
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
