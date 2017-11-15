package main

import (
	"os"
	"time"
	"fmt"
	"math/rand"
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"log"
	"flag"
	"github.com/yaricom/goNEAT/experiments/xor"
)

// The XOR experiment runner
func main() {
	var out_dir_path = flag.String("out", "./out", "The output directory to store results")
	var context_path = flag.String("context", "./data/xor.neat", "The execution context configuration file")
	var genome_path = flag.String("genome", "./data/xorstartgenes", "The seed genome to start with")

	flag.Parse()

	// Seed the random-number generator with current time so that
	// the numbers will be different every time we run.
	rand.Seed(time.Now().Unix())

	// Load context configuration
	configFile, err := os.Open(*context_path)
	if err != nil {
		log.Fatal("Failed to open context configuration file: ", err)
	}
	context := neat.LoadContext(configFile)

	// Load Genome
	log.Println("Loading start genome for XOR experiment")
	genomeFile, err := os.Open(*genome_path)
	if err != nil {
		log.Fatal("Failed to open genome file: ", err)
	}
	start_genome, err := genetics.ReadGenome(genomeFile, 1)
	if err != nil {
		log.Fatal("Failed to read start genome: ", err)
	}
	fmt.Println(start_genome)

	// Check if output dir exists
	if _, err := os.Stat(*out_dir_path); err == nil {
		// clear it
		os.RemoveAll(*out_dir_path)
	}
	// create output dir
	err = os.MkdirAll(*out_dir_path, os.ModePerm)
	if err != nil {
		log.Fatal("Failed to create output directory: ", err)
	}

	// The 100 generation XOR experiment
	experiment := experiments.Experiment {
		Id:0,
		Trials:make(experiments.Trials, context.NumRuns),
	}
	err = experiment.Execute(context, start_genome, xor.XOREpochExecutor{OutputPath:*out_dir_path})
	if err != nil {
		log.Fatal("Failed to perform XOR experiment: ", err)
	}

	// Find winner statistics
	avg_nodes, avg_genes, avg_evals := experiment.AvgWinnerNGE()


	fmt.Printf("\nAverage\n\tWinner Nodes:\t%.1f\n\tWinner Genes:\t%.1f\n\tWinner Evals:\t%.1f\n",
		avg_nodes, avg_genes, avg_evals)
	mean_complexity, mean_diversity, mean_age := 0.0, 0.0, 0.0
	for _, t := range experiment.Trials {
		mean_complexity += t.Complexity().Mean()
		mean_diversity += t.Diversity().Mean()
		mean_age += t.Age().Mean()
	}
	count := float64(len(experiment.Trials))
	mean_complexity /= count
	mean_diversity /= count
	mean_age /= count
	fmt.Printf("Mean\n\tComplexity:\t%.1f\n\tDiversity:\t%.1f\n\tAge:\t\t%.1f\n", mean_complexity, mean_diversity, mean_age)

	fmt.Printf(">>> Start genome file:  %s\n", *genome_path)
	fmt.Printf(">>> Configuration file: %s\n", *context_path)
}