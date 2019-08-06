// The XOR experiment serves to actually check that network topology actually evolves and everything works as expected.
// Because XOR is not linearly separable, a neural network requires hidden units to solve it. The two inputs must be
// combined at some hidden unit, as opposed to only at the out- put node, because there is no function over a linear
// combination of the inputs that can separate the inputs into the proper classes. These structural requirements make
// XOR suitable for testing NEAT’s ability to evolve structure.
package xor

import (
	"github.com/yaricom/goNEAT/neat"
	"os"
	"fmt"
	"github.com/yaricom/goNEAT/neat/genetics"
	"math"
	"github.com/yaricom/goNEAT/experiments"
)

// The fitness threshold value for successful solver
const fitness_threshold = 15.5

// XOR is very simple and does not make a very interesting scientific experiment; however, it is a good way to
// check whether your system works.
// Make sure recurrency is disabled for the XOR test. If NEAT is able to add recurrent connections, it may solve XOR by
// memorizing the order of the training set. (Which is why you may even want to randomize order to be most safe) All
// documented experiments with XOR are without recurrent connections. Interestingly, XOR can be solved by a recurrent
// network with no hidden nodes.
//
// This method performs evolution on XOR for specified number of generations and output results into outDirPath
// It also returns number of nodes, genes, and evaluations performed per each run (context.NumRuns)
type XORGenerationEvaluator struct {
	// The output path to store execution results
	OutputPath string
}

// This method evaluates one epoch for given population and prints results into output directory if any.
func (ex XORGenerationEvaluator) GenerationEvaluate(pop *genetics.Population, epoch *experiments.Generation, context *neat.NeatContext) (err error) {
	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		res, err := ex.org_evaluate(org, context)
		if err != nil {
			return err
		}

		if res && (epoch.Best == nil || org.Fitness > epoch.Best.Fitness){
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize * epoch.Id + org.Genotype.Id
			epoch.Best = org
			if (epoch.WinnerNodes == 5) {
				// You could dump out optimal genomes here if desired
				opt_path := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(ex.OutputPath, epoch.TrialId),
					"xor_optimal", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				file, err := os.Create(opt_path)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
				} else {
					org.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", opt_path))
				}
			}
		}
	}

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(pop)

	// Only print to file every print_every generations
	if epoch.Solved || epoch.Id % context.PrintEvery == 0 {
		pop_path := fmt.Sprintf("%s/gen_%d", experiments.OutDirForTrial(ex.OutputPath, epoch.TrialId), epoch.Id)
		file, err := os.Create(pop_path)
		if err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
		} else {
			pop.WriteBySpecies(file)
		}
	}

	if epoch.Solved {
		// print winner organism
		for _, org := range pop.Organisms {
			if org.IsWinner {
				// Prints the winner organism to file!
				org_path := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(ex.OutputPath, epoch.TrialId),
					"xor_winner", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				file, err := os.Create(org_path)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism genome, reason: %s\n", err))
				} else {
					org.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Generation #%d winner dumped to: %s\n", epoch.Id, org_path))
				}
				break
			}
		}
	}

	return err
}

// This methods evaluates provided organism
func (ex *XORGenerationEvaluator) org_evaluate(organism *genetics.Organism, context *neat.NeatContext) (bool, error) {
	// The four possible input combinations to xor
	// The first number is for biasing
	in := [][]float64{
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0},
		{1.0, 1.0, 1.0}}

	net_depth, err := organism.Phenotype.MaxDepth() // The max depth of the network to be activated
	if err != nil {
		neat.WarnLog(
			fmt.Sprintf("Failed to estimate maximal depth of the network with loop:\n%s\nUsing default dpeth: %d",
				organism.Genotype, net_depth))
	}
	neat.DebugLog(fmt.Sprintf("Network depth: %d for organism: %d\n", net_depth, organism.Genotype.Id))
	if net_depth == 0 {
		neat.DebugLog(fmt.Sprintf("ALERT: Network depth is ZERO for Genome: %s", organism.Genotype))
	}

	success := false  // Check for successful activation
	out := make([]float64, 4) // The four outputs

	// Load and activate the network on each input
	for count := 0; count < 4; count++ {
		organism.Phenotype.LoadSensors(in[count])

		// Relax net and get output
		success, err = organism.Phenotype.Activate()
		if err != nil {
			neat.ErrorLog("Failed to activate network")
			return false, err
		}

		// use depth to ensure relaxation
		for relax := 0; relax <= net_depth; relax++ {
			success, err = organism.Phenotype.Activate()
			if err != nil {
				neat.ErrorLog("Failed to activate network")
				return false, err
			}
		}
		out[count] = organism.Phenotype.Outputs[0].Activation

		organism.Phenotype.Flush()
	}

	if (success) {
		// Mean Squared Error
		error_sum := math.Abs(out[0]) + math.Abs(1.0 - out[1]) + math.Abs(1.0 - out[2]) + math.Abs(out[3]) // ideal == 0
		target := 4.0 - error_sum // ideal == 4.0
		organism.Fitness = math.Pow(4.0 - error_sum, 2.0)
		organism.Error = math.Pow(4.0 - target, 2.0)
	} else {
		// The network is flawed (shouldn't happen) - flag as anomaly
		organism.Error = 1.0
		organism.Fitness = 0.0
	}

	if organism.Fitness > fitness_threshold {
		organism.IsWinner = true
		neat.InfoLog(fmt.Sprintf(">>>> Output activations: %e\n", out))

	} else {
		organism.IsWinner = false
	}
	return organism.IsWinner, nil
}