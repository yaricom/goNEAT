package experiments

import (
	"github.com/yaricom/goNEAT/neat"
	"os"
	"fmt"
	"github.com/yaricom/goNEAT/neat/genetics"
	"math"
)

// XOR is very simple and does not make a very interesting scientific experiment; however, it is a good way to
// check whether your system works.
// Make sure recurrency is disabled for the XOR test. If NEAT is able to add recurrent connections, it may solve XOR by
// memorizing the order of the training set. (Which is why you may even want to randomize order to be most safe) All
// documented experiments with XOR are without recurrent connections. Interestingly, XOR can be solved by a recurrent
// network with no hidden nodes.
//
// This method performs evolution on XOR for specified number of generations. It will read NEAT context configuration
// from contextPath, the start genome configuration from genomePath, and output results into outDirPath
func XOR(context_path, genome_path, out_dir_path string, generations int) (*genetics.Population, error) {

	// Load context configuration
	configFile, err := os.Open(context_path)
	if err != nil {
		fmt.Println("Failed to load context")
		return nil, err
	}
	context := neat.LoadContext(configFile)
	context.IsDebugEnabled = true

	// Load Genome
	fmt.Println("Loading start genome for XOR experiment")
	genomeFile, err := os.Open(genome_path)
	if err != nil {
		fmt.Println("Failed to open genome file")
		return nil, err
	}
	start_genome, err := genetics.ReadGenome(genomeFile, 1)
	if err != nil {
		fmt.Println("Failed to read start genome")
		return nil, err
	}
	fmt.Println(start_genome)

	// Check if output dir exists
	if _, err := os.Stat(out_dir_path); os.IsNotExist(err) {
		// create output dir
		err = os.MkdirAll(out_dir_path, os.ModePerm)
		if err != nil {
			return nil, err
		}
	}


	// Holders of records for each run
	evals := make([]int, context.NumRuns)
	genes := make([]int, context.NumRuns)
	nodes := make([]int, context.NumRuns)

	var successful_pop *genetics.Population

	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		fmt.Print("\n>>>>> Spawning new population ")
		pop, err := genetics.NewPopulation(start_genome, context)
		if err != nil {
			fmt.Println("Failed to spawn new population from start genome")
			return nil, err
		} else {
			fmt.Println("OK <<<<<")
		}
		fmt.Print(">>>>> Verifying spawned population ")
		_, err = pop.Verify()
		if err != nil {
			fmt.Println("\n!!!!! Population verification failed !!!!!")
			return nil, err
		} else {
			fmt.Println("OK <<<<<\n")
		}

		for gen := 1; gen <= generations; gen ++ {
			fmt.Printf(">>>>> Epoch: %d\n", gen)
			success, winner_num, winner_genes, winner_nodes, err := xor_epoch(pop, gen, out_dir_path, context)
			if err != nil {
				fmt.Println("!!!!! Epoch evaluation failed !!!!!")
				return nil, err
			}
			if success {
				// Collect Stats on end of experiment
				evals[exp_count] = context.PopSize * (gen - 1) + winner_num
				genes[exp_count] = winner_genes
				nodes[exp_count] = winner_nodes
				fmt.Println(">>>>> The winner organism found! <<<<<")
				break
			}
		}

	}

	// Average and print stats

	fmt.Print("\n>>>>> Nodes: ")
	total_nodes := 0
	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		fmt.Printf("\t%d",nodes[exp_count])
		total_nodes += nodes[exp_count]
	}

	fmt.Print("\n>>>>> Genes: ")
	total_genes := 0
	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		fmt.Printf("\t%d", genes[exp_count])
		total_genes += genes[exp_count]
	}

	fmt.Print("\n>>>>> Evals: ")
	total_evals := 0
	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		fmt.Printf("\t%d", evals[exp_count])
		total_evals += evals[exp_count]
	}

	fmt.Printf("\n>>>>>\nAverage Nodes:\t%d\nAverage Genes:\t%d\nAverage Evals:\t%d\n",
		total_nodes / context.NumRuns, total_genes / context.NumRuns, total_evals / context.NumRuns)

	return successful_pop, nil
}

// This method evaluates one epoch for given population and prints results into specified directory if any.
func xor_epoch(pop *genetics.Population, generation int, out_dir_path string, context *neat.NeatContext) (success bool, winner_num, winner_genes, winner_nodes int, err error) {
	// The flag to indicate that we have winner organism
	success = false
	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		res, err := xor_evaluate(org)
		if err != nil {
			return false, -1, -1, -1, err
		}
		if res {
			success = true
			winner_num = org.GNome.Id
			winner_genes = org.GNome.Extrons()
			winner_nodes = len(org.GNome.Nodes)
			if (winner_nodes == 5) {
				// You could dump out optimal genomes here if desired
				opt_path := fmt.Sprintf("%s/%s", out_dir_path, "xor_optimal")
				file, err := os.Create(opt_path)
				if err != nil {
					fmt.Printf("Failed to dump optimal genome, reason: %s\n", err)
				} else {
					org.GNome.Write(file)
					fmt.Printf("Dumped optimal genome to: %s\n", opt_path)
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
			fmt.Printf("Failed to dump population, reason: %s\n", err)
		} else {
			pop.WriteBySpecies(file)
		}
	}

	if success {
		for _, org := range pop.Organisms {
			if org.IsWinner {
				// Prints the winner organism to file!
				org_path := fmt.Sprintf("%s/%s", out_dir_path, "xor_winner")
				file, err := os.Create(org_path)
				if err != nil {
					fmt.Printf("Failed to dump winner organism genome, reason: %s\n", err)
				} else {
					org.GNome.Write(file)
					fmt.Printf("Generation #%d winner dumped to: %s\n", generation, org_path)
				}
				break
			}
		}
	}

	// Move to the next epoch
	fmt.Println(">>>>> start next generation")
	_, err = pop.Epoch(generation, context)

	return success, winner_num, winner_genes, winner_nodes, err
}

// This methods evalueates provided organism
func xor_evaluate(org *genetics.Organism) (bool, error) {
	// The four possible input combinations to xor
	// The first number is for biasing
	in := [][]float64{
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0},
		{1.0, 1.0, 1.0}}

	net_depth, err := org.Net.MaxDepth() // The max depth of the network to be activated
	if err != nil {
		fmt.Println("Failed to estimate maximal depth of the network")
		return false, err
	}

	success := false  // Check for successful activation
	out := make([]float64, 4) // The four outputs

	// Load and activate the network on each input
	for count := 0; count < 3; count++ {
		org.Net.LoadSensors(in[count])

		// Relax net and get output
		success, err = org.Net.Activate()
		if err != nil {
			fmt.Println("Failed to activate network")
			return false, err
		}

		// use depth to ensure relaxation
		for relax := 0; relax <= net_depth; relax++ {
			success, err = org.Net.Activate()
			if err != nil {
				fmt.Println("Failed to activate network")
				return false, err
			}
		}
		out[count] = org.Net.Outputs[0].Activation

		fmt.Println(org.Net.Outputs)

		org.Net.Flush()
	}

	error_sum := 0.0
	if (success) {
		// Mean Squared Error
		error_sum = math.Abs(out[0]) + math.Abs(1.0 - out[1]) + math.Abs(1.0 - out[2]) + math.Abs(out[3])
		org.Fitness = math.Pow(4.0 - error_sum, 2.0)
		org.Error = error_sum
	} else {
		// The network is flawed (shouldn't happen)
		error_sum = 999.0
		org.Fitness = 0.001
	}

	if out[0] < 0.5 && out[1] >= 0.5 && out[2] >= 0.5 && out[3] < 0.5 {
		org.IsWinner = true
	} else {
		org.IsWinner = false
	}
	return org.IsWinner, nil
}
