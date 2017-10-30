package xor

import (
	"testing"
	"time"
	"os"
	"fmt"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"math/rand"
	"github.com/yaricom/goNEAT/experiments"
)

// The integration test running over multiple iterations in order to detect if any random errors occur.
func TestXOR(t *testing.T) {
	// the numbers will be different every time we run.
	rand.Seed(time.Now().Unix())

	out_dir_path, context_path, genome_path := "../../out", "../../data/xor.neat", "../../data/xorstartgenes"

	// Load context configuration
	configFile, err := os.Open(context_path)
	if err != nil {
		t.Error("Failed to load context", err)
		return
	}
	context := neat.LoadContext(configFile)
	neat.LogLevel = neat.LogLevelInfo

	// Load Genome
	fmt.Println("Loading start genome for XOR experiment")
	genomeFile, err := os.Open(genome_path)
	if err != nil {
		t.Error("Failed to open genome file")
		return
	}
	start_genome, err := genetics.ReadGenome(genomeFile, 1)
	if err != nil {
		t.Error("Failed to read start genome")
		return
	}

	// Check if output dir exists
	if _, err := os.Stat(out_dir_path); err == nil {
		// clear it
		os.RemoveAll(out_dir_path)
	}
	// create output dir
	err = os.MkdirAll(out_dir_path, os.ModePerm)
	if err != nil {
		t.Errorf("Failed to create output directory, reason: %s", err)
		return
	}

	// The 100 runs XOR experiment
	context.NumRuns = 100
	experiment := experiments.Experiment {
		Id:0,
		Trials:make(experiments.Trials, context.NumRuns),
	}
	nodes, genes, evals, err := XOR(context, start_genome, out_dir_path, &experiment)
	if err != nil {
		t.Error("Failed to perform XOR experiment:", err)
		return
	}

	// Find statistics
	total_nodes := 0
	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		total_nodes += nodes[exp_count]
	}

	total_genes := 0
	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		total_genes += genes[exp_count]
	}

	total_evals := 0
	for exp_count := 0; exp_count < context.NumRuns; exp_count++ {
		total_evals += evals[exp_count]
	}

	// check results
	avg_nodes := float64(total_nodes) / float64(context.NumRuns)
	if avg_nodes < 5 {
		t.Error("avg_nodes < 5", avg_nodes)
	} else if avg_nodes > 15 {
		t.Error("avg_nodes > 15", avg_nodes)
	}

	avg_genes := float64(total_genes) / float64(context.NumRuns)
	if avg_genes < 7 {
		t.Error("avg_genes < 7", avg_genes)
	} else if avg_genes > 20 {
		t.Error("avg_genes > 20", avg_genes)
	}

	avg_evals := total_evals / context.NumRuns
	max_evals := context.NumRuns * context.NumGenerations
	if avg_evals > max_evals {
		t.Error("avg_evals > max_evals", avg_evals, max_evals)
	}

	t.Logf("avg_nodes: %.1f, avg_genes: %.1f, avg_evals: %d", avg_nodes, avg_genes, avg_evals)
}
