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
	"time"
)

// The integration test running running over multiple iterations
func TestCartPoleGenerationEvaluator_GenerationEvaluate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short Unit Test mode.")
	}

	// the numbers will be different every time we run.
	rand.Seed(time.Now().Unix())

	outDirPath, contextPath, genomePath := "../../out/pole1_test", "../../data/pole1_1000.neat", "../../data/pole1startgenes"

	fmt.Println("Loading start genome for POLE1 experiment")

	// Load NEAT options and initial genome
	opts, startGenome, err := utils.LoadOptionsAndGenome(contextPath, genomePath)
	neat.LogLevel = neat.LogLevelInfo
	require.NoError(t, err)

	// Check if output dir exists
	err = utils.CreateOutputDir(outDirPath)
	require.NoError(t, err, "Failed to create output directory")

	// The 100 runs POLE1 experiment
	opts.NumRuns = 100
	experiment := experiment2.Experiment{
		Id:     0,
		Trials: make(experiment2.Trials, opts.NumRuns),
	}
	evaluator := NewCartPoleGenerationEvaluator(outDirPath, true, 500000)
	err = experiment.Execute(opts.NeatContext(), startGenome, evaluator, nil)
	require.NoError(t, err, "Failed to perform POLE1 experiment")

	// Find winner statistics
	avgNodes, avgGenes, avgEvals, _ := experiment.AvgWinner()

	// check results
	if avgNodes < 7 {
		t.Error("avg_nodes < 7", avgNodes)
	} else if avgNodes > 10 {
		t.Error("avg_nodes > 10", avgNodes)
	}

	if avgGenes < 10 {
		t.Error("avg_genes < 10", avgGenes)
	} else if avgGenes > 20 {
		t.Error("avg_genes > 20", avgGenes)
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
