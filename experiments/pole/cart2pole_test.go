package pole

import (
	"testing"
	"os"
	"github.com/yaricom/goNEAT/neat"
	"fmt"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/experiments"
	"math/rand"
)

// Run double pole-balancing experiment with Markov environment setup
func TestCartDoublePoleGenerationEvaluator_GenerationEvaluateMarkov(t *testing.T) {
	// to make sure we have predictable results
	rand.Seed(3423)

	out_dir_path, context_path, genome_path := "../../out/pole2_markov_test", "../../data/pole2_markov.neat", "../../data/pole2_markov_startgenes"

	// Load context configuration
	configFile, err := os.Open(context_path)
	if err != nil {
		t.Error("Failed to load context", err)
		return
	}
	context := neat.LoadContext(configFile)
	neat.LogLevel = neat.LogLevelInfo

	// Load Genome
	fmt.Println("Loading start genome for POLE2 Markov experiment")
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

	// The 10 runs POLE2 Markov experiment
	context.NumRuns = 5
	experiment := experiments.Experiment{
		Id:0,
		Trials:make(experiments.Trials, context.NumRuns),
	}
	err = experiment.Execute(context, start_genome, CartDoublePoleGenerationEvaluator{
		OutputPath:out_dir_path,
		Markov:true,
		ActionType:experiments.ContinuousAction,
	})
	if err != nil {
		t.Error("Failed to perform POLE2 Markov experiment:", err)
		return
	}

	// Find winner statistics
	avg_nodes, avg_genes, avg_evals, _ := experiment.AvgWinner()

	// check results
	if avg_nodes < 8 {
		t.Error("avg_nodes < 8", avg_nodes)
	} else if avg_nodes > 40 {
		t.Error("avg_nodes > 40", avg_nodes)
	}

	if avg_genes < 7 {
		t.Error("avg_genes < 7", avg_genes)
	} else if avg_genes > 50 {
		t.Error("avg_genes > 50", avg_genes)
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

// Run double pole-balancing experiment with Non-Markov environment setup
func TestCartDoublePoleGenerationEvaluator_GenerationEvaluateNonMarkov(t *testing.T) {
	// to make sure we have predictable results
	rand.Seed(423)

	out_dir_path, context_path, genome_path := "../../out/pole2_non-markov_test", "../../data/pole2_non-markov.neat", "../../data/pole2_non-markov_startgenes"

	// Load context configuration
	configFile, err := os.Open(context_path)
	if err != nil {
		t.Error("Failed to load context", err)
		return
	}
	context := neat.LoadContext(configFile)
	neat.LogLevel = neat.LogLevelInfo

	// Load Genome
	fmt.Println("Loading start genome for POLE2 Non-Markov experiment")
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

	// The 10 runs POLE2 Non-Markov experiment
	context.NumRuns = 5
	experiment := experiments.Experiment{
		Id:0,
		Trials:make(experiments.Trials, context.NumRuns),
	}
	err = experiment.Execute(context, start_genome, CartDoublePoleGenerationEvaluator{
		OutputPath:out_dir_path,
		Markov:false,
		ActionType:experiments.ContinuousAction,
	})
	if err != nil {
		t.Error("Failed to perform POLE2 Non-Markov experiment:", err)
		return
	}

	// Find winner statistics
	avg_nodes, avg_genes, avg_evals, _ := experiment.AvgWinner()

	// check results
	if avg_nodes < 5 {
		t.Error("avg_nodes < 5", avg_nodes)
	} else if avg_nodes > 40 {
		t.Error("avg_nodes > 40", avg_nodes)
	}

	if avg_genes < 5 {
		t.Error("avg_genes < 5", avg_genes)
	} else if avg_genes > 50 {
		t.Error("avg_genes > 50", avg_genes)
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
	t.Logf("Trials solved/run: %d/%d\n", solved_trials, len(experiment.Trials))

	if solved_trials == 0 {
		t.Error("Failed to solve at least one trial. Need to be checked what was going wrong")
	}

	best_g_score := 0.0
	for _, tr := range experiment.Trials {
		if org, found := tr.BestOrganism(true); found {
			best_org_score := org.Fitness
			if best_org_score > best_g_score {
				best_g_score = best_org_score
			}
		}
	}
	t.Logf("Best Generalization Score: %.0f\n", best_g_score)
}
