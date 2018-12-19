package pole

import (
	"testing"
	"math/rand"
	"time"
	"os"
	"fmt"
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
)

// The integration test running running over multiple iterations
func TestCartPoleGenerationEvaluator_GenerationEvaluate(t *testing.T) {
	// the numbers will be different every time we run.
	rand.Seed(time.Now().Unix())

	out_dir_path, context_path, genome_path := "../../out/pole1_test", "../../data/pole1_1000.neat", "../../data/pole1startgenes"

	// Load context configuration
	configFile, err := os.Open(context_path)
	if err != nil {
		t.Error("Failed to load context", err)
		return
	}
	context := neat.LoadContext(configFile)
	neat.LogLevel = neat.LogLevelInfo

	// Load Genome
	fmt.Println("Loading start genome for POLE1 experiment")
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

	// The 100 runs POLE1 experiment
	context.NumRuns = 100
	experiment := experiments.Experiment {
		Id:0,
		Trials:make(experiments.Trials, context.NumRuns),
	}
	err = experiment.Execute(context, start_genome, CartPoleGenerationEvaluator{
		OutputPath:out_dir_path,
		WinBalancingSteps:500000,
		RandomStart:true,
	})
	if err != nil {
		t.Error("Failed to perform POLE1 experiment:", err)
		return
	}

	// Find winner statistics
	avg_nodes, avg_genes, avg_evals, _ := experiment.AvgWinner()

	// check results
	if avg_nodes < 7 {
		t.Error("avg_nodes < 7", avg_nodes)
	} else if avg_nodes > 10 {
		t.Error("avg_nodes > 10", avg_nodes)
	}

	if avg_genes < 10 {
		t.Error("avg_genes < 10", avg_genes)
	} else if avg_genes > 20 {
		t.Error("avg_genes > 20", avg_genes)
	}

	max_evals := float64(context.PopSize * context.NumGenerations)
	if avg_evals > max_evals {
		t.Error("avg_evals > max_evals", avg_evals, max_evals)
	}

	t.Logf("Average nodes: %.1f, genes: %.1f, evals: %.1f\n", avg_nodes, avg_genes, avg_evals)
	mean_complexity, mean_diversity, mean_age := 0.0, 0.0, 0.0
	for _, t := range experiment.Trials {
		mean_complexity += t.BestComplexity().Mean()
		mean_diversity += t.Diversity().Mean()
		mean_age += t.BestAge().Mean()
	}
	count := float64(len(experiment.Trials))
	mean_complexity /= count
	mean_diversity /= count
	mean_age /= count
	t.Logf("Mean best organisms: complexity=%.1f, diversity=%.1f, age=%.1f\n", mean_complexity, mean_diversity, mean_age)

	solved_trials := 0
	for _, tr := range experiment.Trials {
		if tr.Solved() {
			solved_trials++
		}
	}

	t.Logf("Trials solved/run: %d/%d", solved_trials, len(experiment.Trials))

	if solved_trials == 0 {
		t.Error("Failed to solve at least one trial. Need to be checked what was going wrong")
	}
}
