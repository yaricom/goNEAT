package main

import (
	"os"
	"time"
	"fmt"
	"math/rand"
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
)

// The XOR experiment runner
func main() {
	out_dir_path, context_path, genome_path := "./out", "./data/xor.neat", "./data/xorstartgenes"
	if len(os.Args) == 4 {
		out_dir_path = os.Args[1]
		context_path = os.Args[2]
		genome_path = os.Args[3]
	}

	// Seed the random-number generator with current time so that
	// the numbers will be different every time we run.
	rand.Seed(time.Now().Unix())

	// Load context configuration
	configFile, err := os.Open(context_path)
	if err != nil {
		fmt.Println("Failed to load context")
		return
	}
	context := neat.LoadContext(configFile)

	// Load Genome
	fmt.Println("Loading start genome for XOR experiment")
	genomeFile, err := os.Open(genome_path)
	if err != nil {
		fmt.Println("Failed to open genome file")
		return
	}
	start_genome, err := genetics.ReadGenome(genomeFile, 1)
	if err != nil {
		fmt.Println("Failed to read start genome")
		return
	}
	fmt.Println(start_genome)

	// Check if output dir exists
	if _, err := os.Stat(out_dir_path); err == nil {
		// clear it
		os.RemoveAll(out_dir_path)
	}
	// create output dir
	err = os.MkdirAll(out_dir_path, os.ModePerm)
	if err != nil {
		fmt.Printf("Failed to create output directory, reason: %s", err)
		return
	}

	// The 100 generation XOR experiment
	nodes, genes, evals, err := experiments.XOR(context, start_genome, out_dir_path)
	if err != nil {
		fmt.Println("Failed to perform XOR experiment:")
		fmt.Println(err)
		return
	}

	// Average and print stats
	fmt.Print("\nNodes: ")
	total_nodes := 0
	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		fmt.Printf("\t%d", nodes[exp_count])
		total_nodes += nodes[exp_count]
	}

	fmt.Print("\nGenes: ")
	total_genes := 0
	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		fmt.Printf("\t%d", genes[exp_count])
		total_genes += genes[exp_count]
	}

	fmt.Print("\nEvals: ")
	total_evals := 0
	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		fmt.Printf("\t%d", evals[exp_count])
		total_evals += evals[exp_count]
	}

	fmt.Printf("\n>>>>>\nAverage Nodes:\t%d\nAverage Genes:\t%d\nAverage Evals:\t%d\n",
		total_nodes / context.NumRuns, total_genes / context.NumRuns, total_evals / context.NumRuns)

	fmt.Printf(">>> Start genome file:  %s\n", genome_path)
	fmt.Printf(">>> Configuration file: %s\n", context_path)
}