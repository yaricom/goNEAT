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
	err = XOR(context, start_genome, out_dir_path, &experiment)
	if err != nil {
		t.Error("Failed to perform XOR experiment:", err)
		return
	}

	// Find winner statistics
	avg_nodes, avg_genes, avg_evals := experiment.AvgWinnerNGE()

	// check results
	if avg_nodes < 5 {
		t.Error("avg_nodes < 5", avg_nodes)
	} else if avg_nodes > 15 {
		t.Error("avg_nodes > 15", avg_nodes)
	}

	if avg_genes < 7 {
		t.Error("avg_genes < 7", avg_genes)
	} else if avg_genes > 20 {
		t.Error("avg_genes > 20", avg_genes)
	}

	max_evals := float64(context.PopSize * context.NumGenerations)
	if avg_evals > max_evals {
		t.Error("avg_evals > max_evals", avg_evals, max_evals)
	}

	t.Logf("avg_nodes: %.1f, avg_genes: %.1f, avg_evals: %.1f\n", avg_nodes, avg_genes, avg_evals)
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
	t.Logf("mean: complexity=%.1f, diversity=%.1f, age=%.1f", mean_complexity, mean_diversity, mean_age)
}


// The XOR integration test for disconnected inputs running over multiple iterations in order to detect if any random errors occur.
func TestXOR_disconnected(t *testing.T) {
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
	err = XOR(context, start_genome, out_dir_path, &experiment)
	if err != nil {
		t.Error("Failed to perform XOR experiment:", err)
		return
	}

	// Find winner statistics
	avg_nodes, avg_genes, avg_evals := experiment.AvgWinnerNGE()

	// check results
	if avg_nodes < 5 {
		t.Error("avg_nodes < 5", avg_nodes)
	} else if avg_nodes > 15 {
		t.Error("avg_nodes > 15", avg_nodes)
	}

	if avg_genes < 7 {
		t.Error("avg_genes < 7", avg_genes)
	} else if avg_genes > 20 {
		t.Error("avg_genes > 20", avg_genes)
	}

	max_evals := float64(context.PopSize * context.NumGenerations)
	if avg_evals > max_evals {
		t.Error("avg_evals > max_evals", avg_evals, max_evals)
	}

	t.Logf("avg_nodes: %.1f, avg_genes: %.1f, avg_evals: %.1f\n", avg_nodes, avg_genes, avg_evals)
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
	t.Logf("mean: complexity=%.1f, diversity=%.1f, age=%.1f", mean_complexity, mean_diversity, mean_age)
}