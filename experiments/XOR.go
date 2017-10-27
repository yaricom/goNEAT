package experiments

import (
	"github.com/yaricom/goNEAT/neat"
	"os"
	"fmt"
	"github.com/yaricom/goNEAT/neat/genetics"
	"math"
)

// The precision to use for XOR evaluation, i.e. one is x > 1 - precision and zero is x < precision
const precision = 0.5

// XOR is very simple and does not make a very interesting scientific experiment; however, it is a good way to
// check whether your system works.
// Make sure recurrency is disabled for the XOR test. If NEAT is able to add recurrent connections, it may solve XOR by
// memorizing the order of the training set. (Which is why you may even want to randomize order to be most safe) All
// documented experiments with XOR are without recurrent connections. Interestingly, XOR can be solved by a recurrent
// network with no hidden nodes.
//
// This method performs evolution on XOR for specified number of generations and output results into outDirPath
// It also returns number of nodes, genes, and evaluations performed per each run (context.NumRuns)
func XOR(context *neat.NeatContext, start_genome *genetics.Genome, out_dir_path string) (nodes, genes, evals []int, err error) {

	// Holders of records for each run
	evals = make([]int, context.NumRuns)
	genes = make([]int, context.NumRuns)
	nodes = make([]int, context.NumRuns)

	var pop *genetics.Population
	for run := 0; run < context.NumRuns; run++ {
		neat.InfoLog("\n>>>>> Spawning new population ")
		pop, err = genetics.NewPopulation(start_genome, context)
		if err != nil {
			neat.InfoLog("Failed to spawn new population from start genome")
			return nodes, genes, evals, err
		} else {
			neat.InfoLog("OK <<<<<")
		}
		neat.InfoLog(">>>>> Verifying spawned population ")
		_, err = pop.Verify()
		if err != nil {
			neat.ErrorLog("\n!!!!! Population verification failed !!!!!")
			return nodes, genes, evals, err
		} else {
			neat.InfoLog("OK <<<<<")
		}

		var success bool
		var winner_num, winner_genes, winner_nodes int
		for gen := 1; gen <= context.NumGenerations; gen ++ {
			neat.InfoLog(fmt.Sprintf(">>>>> Epoch: %d\tRun: %d\n", gen, run))
			success, winner_num, winner_genes, winner_nodes, err = xor_epoch(pop, gen, out_dir_path, context)
			if err != nil {
				neat.InfoLog(fmt.Sprintf("!!!!! Epoch %d evaluation failed !!!!!\n", gen))
				return nodes, genes, evals, err
			}
			if success {
				// Collect Stats on end of experiment
				evals[run] = context.PopSize * (gen - 1) + winner_num
				genes[run] = winner_genes
				nodes[run] = winner_nodes
				neat.InfoLog(fmt.Sprintf(">>>>> The winner organism found in epoch %d! <<<<<\n", gen))
				break
			}
		}

	}

	return nodes, genes, evals, nil
}

// This method evaluates one epoch for given population and prints results into specified directory if any.
func xor_epoch(pop *genetics.Population, generation int, out_dir_path string, context *neat.NeatContext) (success bool, winner_num, winner_genes, winner_nodes int, err error) {
	// The flag to indicate that we have winner organism
	success = false
	// The best organism and it's fintess
	//var best_organism genetics.Organism
	//max_fitness := 0.0
	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		res, err := xor_evaluate(org, context)
		if err != nil {
			return false, -1, -1, -1, err
		}
		//if org.Fitness > max_fitness {
		//	// store for epoch statistics
		//	max_fitness = org.Fitness
		//	best_organism = org
		//}

		if res {
			success = true
			winner_num = org.Genotype.Id
			winner_genes = org.Genotype.Extrons()
			winner_nodes = len(org.Genotype.Nodes)
			if (winner_nodes == 5) {
				// You could dump out optimal genomes here if desired
				opt_path := fmt.Sprintf("%s/%s", out_dir_path, "xor_optimal")
				file, err := os.Create(opt_path)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
				} else {
					org.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", opt_path))
				}
			}
			break // we have winner
		}
	}
	// Average and max their fitnesses for dumping to file and snapshot
	for _, curr_species := range pop.Species {
		// This experiment control routine issues commands to collect ave and max fitness, as opposed to having
		// the snapshot do it, because this allows flexibility in terms of what time to observe fitnesses at
		curr_species.ComputeAvgFitness()
		curr_species.ComputeMaxFitness()
	}

	// Only print to file every print_every generations
	if success || generation % context.PrintEvery == 0 {
		pop_path := fmt.Sprintf("%s/gen_%d", out_dir_path, generation)
		file, err := os.Create(pop_path)
		if err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
		} else {
			pop.WriteBySpecies(file)
		}
	}

	if success {
		// print winner organism
		for _, org := range pop.Organisms {
			if org.IsWinner {
				// Prints the winner organism to file!
				org_path := fmt.Sprintf("%s/%s", out_dir_path, "xor_winner")
				file, err := os.Create(org_path)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism genome, reason: %s\n", err))
				} else {
					org.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Generation #%d winner dumped to: %s\n", generation, org_path))
				}
				break
			}
		}
	} else {
		// Move to the next epoch if failed to find winner
		neat.DebugLog(">>>>> start next generation")
		_, err = pop.Epoch(generation, context)
	}

	return success, winner_num, winner_genes, winner_nodes, err
}

// This methods evalueates provided organism
func xor_evaluate(organism *genetics.Organism, context *neat.NeatContext) (bool, error) {
	// The four possible input combinations to xor
	// The first number is for biasing
	in := [][]float64{
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0},
		{1.0, 1.0, 1.0}}

	net_depth, err := organism.Phenotype.MaxDepth() // The max depth of the network to be activated
	if err != nil {
		neat.ErrorLog(fmt.Sprintf("Failed to estimate maximal depth of the network with genome:\n%s", organism.Genotype))
		return false, err
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

	error_sum := 0.0
	if (success) {
		// Mean Squared Error
		error_sum = math.Abs(out[0]) + math.Abs(1.0 - out[1]) + math.Abs(1.0 - out[2]) + math.Abs(out[3])
		organism.Fitness = math.Pow(4.0 - error_sum, 2.0)
		organism.Error = error_sum
	} else {
		// The network is flawed (shouldn't happen)
		error_sum = 999.0
		organism.Fitness = 0.001
	}

	if out[0] < precision && out[1] >= 1 - precision && out[2] >= 1 - precision && out[3] < precision {
		organism.IsWinner = true
		neat.InfoLog(fmt.Sprintf(">>>> Output activations: %e\n", out))

	} else {
		organism.IsWinner = false
	}
	return organism.IsWinner, nil
}
