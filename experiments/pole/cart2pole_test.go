package pole

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	experiment2 "github.com/yaricom/goNEAT/v2/experiment"
	"github.com/yaricom/goNEAT/v2/experiments/utils"
	"github.com/yaricom/goNEAT/v2/neat"
	"math/rand"
	"testing"
)

// Run double pole-balancing experiment with Markov environment setup
func TestCartDoublePoleGenerationEvaluator_GenerationEvaluateMarkov(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short Unit Test mode.")
	}

	// to make sure we have predictable results
	rand.Seed(423)

	outDirPath, contextPath, genomePath := "../../out/pole2_markov_test", "../../data/pole2_markov.neat", "../../data/pole2_markov_startgenes"

	fmt.Println("Loading start genome for POLE2 Markov experiment")
	// Load NEAT options and initial genome
	opts, startGenome, err := utils.LoadOptionsAndGenome(contextPath, genomePath)
	neat.LogLevel = neat.LogLevelInfo
	require.NoError(t, err)

	// Check if output dir exists
	err = utils.CreateOutputDir(outDirPath)
	require.NoError(t, err, "Failed to create output directory")

	// Running POLE2 Markov experiment
	opts.NumRuns = 5
	experiment := experiment2.Experiment{
		Id:     0,
		Trials: make(experiment2.Trials, opts.NumRuns),
	}
	err = experiment.Execute(opts.NeatContext(), startGenome, NewCartDoublePoleGenerationEvaluator(outDirPath, true, ContinuousAction))
	require.NoError(t, err, "Failed to perform POLE2 Markov experiment")

	// Find winner statistics
	avgNodes, avgGenes, avgEvals, _ := experiment.AvgWinner()

	// check results
	if avgNodes < 8 {
		t.Error("avg_nodes < 8", avgNodes)
	} else if avgNodes > 40 {
		t.Error("avg_nodes > 40", avgNodes)
	}

	if avgGenes < 7 {
		t.Error("avg_genes < 7", avgGenes)
	} else if avgGenes > 50 {
		t.Error("avg_genes > 50", avgGenes)
	}

	maxEvals := float64(opts.PopSize * opts.NumGenerations)
	assert.True(t, avgEvals < maxEvals)

	t.Logf("Average nodes: %.1f, genes: %.1f, evals: %.1f\n", avgNodes, avgGenes, avgEvals)
	meanComplexity, meanDiversity, meanAge := 0.0, 0.0, 0.0
	for _, t := range experiment.Trials {
		meanComplexity += t.BestComplexity().Mean()
		meanDiversity += t.Diversity().Mean()
		meanAge += t.BestAge().Mean()
	}
	count := float64(len(experiment.Trials))
	meanComplexity /= count
	meanDiversity /= count
	meanAge /= count
	t.Logf("Mean best organisms: complexity=%.1f, diversity=%.1f, age=%.1f\n", meanComplexity, meanDiversity, meanAge)

	solvedTrials := 0
	for _, tr := range experiment.Trials {
		if tr.Solved() {
			solvedTrials++
		}
	}

	t.Logf("Trials solved/run: %d/%d", solvedTrials, len(experiment.Trials))

	assert.NotZero(t, solvedTrials, "Failed to solve at least one trial. Need to be checked what was going wrong")
}

// Run double pole-balancing experiment with Non-Markov environment setup
func TestCartDoublePoleGenerationEvaluator_GenerationEvaluateNonMarkov(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short Unit Test mode.")
	}

	// to make sure we have predictable results
	rand.Seed(423)

	outDirPath, contextPath, genomePath := "../../out/pole2_non-markov_test", "../../data/pole2_non-markov.neat", "../../data/pole2_non-markov_startgenes"

	fmt.Println("Loading start genome for POLE2 Non-Markov experiment")
	// Load NEAT options and initial genome
	opts, startGenome, err := utils.LoadOptionsAndGenome(contextPath, genomePath)
	neat.LogLevel = neat.LogLevelInfo
	require.NoError(t, err)

	// Check if output dir exists
	err = utils.CreateOutputDir(outDirPath)
	require.NoError(t, err, "Failed to create output directory")

	// Running POLE2 Non-Markov experiment
	opts.NumRuns = 5
	experiment := experiment2.Experiment{
		Id:     0,
		Trials: make(experiment2.Trials, opts.NumRuns),
	}
	err = experiment.Execute(opts.NeatContext(), startGenome, NewCartDoublePoleGenerationEvaluator(outDirPath, false, ContinuousAction))
	require.NoError(t, err, "Failed to perform POLE2 Non-Markov experiment")

	// Find winner statistics
	avgNodes, avgGenes, avgEvals, _ := experiment.AvgWinner()

	// check results
	if avgNodes < 5 {
		t.Error("avg_nodes < 5", avgNodes)
	} else if avgNodes > 40 {
		t.Error("avg_nodes > 40", avgNodes)
	}

	if avgGenes < 5 {
		t.Error("avg_genes < 5", avgGenes)
	} else if avgGenes > 50 {
		t.Error("avg_genes > 50", avgGenes)
	}

	maxEvals := float64(opts.PopSize * opts.NumGenerations)
	assert.True(t, avgEvals < maxEvals)

	t.Logf("Average nodes: %.1f, genes: %.1f, evals: %.1f\n", avgNodes, avgGenes, avgEvals)
	meanComplexity, meanDiversity, meanAge := 0.0, 0.0, 0.0
	for _, t := range experiment.Trials {
		meanComplexity += t.BestComplexity().Mean()
		meanDiversity += t.Diversity().Mean()
		meanAge += t.BestAge().Mean()
	}
	count := float64(len(experiment.Trials))
	meanComplexity /= count
	meanDiversity /= count
	meanAge /= count
	t.Logf("Mean best organisms: complexity=%.1f, diversity=%.1f, age=%.1f\n", meanComplexity, meanDiversity, meanAge)

	solvedTrials := 0
	for _, tr := range experiment.Trials {
		if tr.Solved() {
			solvedTrials++
		}
	}
	t.Logf("Trials solved/run: %d/%d\n", solvedTrials, len(experiment.Trials))

	require.NotZero(t, solvedTrials, "Failed to solve at least one trial. Need to be checked what was going wrong")

	bestGeneralizationScore := 0.0
	for _, tr := range experiment.Trials {
		if org, found := tr.BestOrganism(true); found {
			bestOrgScore := org.Fitness
			if bestOrgScore > bestGeneralizationScore {
				bestGeneralizationScore = bestOrgScore
			}
		}
	}
	t.Logf("Best Generalization Score: %.0f\n", bestGeneralizationScore)
}
