package pole2

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/examples/utils"
	"github.com/yaricom/goNEAT/v4/experiment"
	"github.com/yaricom/goNEAT/v4/neat"
	"math/rand"
	"testing"
)

func TestCartDoublePoleParallelGenerationEvaluator_GenerationEvaluate_Markov(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short Unit Test mode.")
	}

	// to make sure we have predictable results
	rand.Seed(423)

	outDirPath, contextPath, genomePath := "../../out/pole2_markov_parallel_test", "../../data/pole2_markov.neat", "../../data/pole2_markov_startgenes"

	fmt.Println("Loading start genome for POLE2 Markov parallel experiment")
	// Load NEAT options and initial genome
	opts, startGenome, err := utils.LoadOptionsAndGenome(contextPath, genomePath)
	neat.LogLevel = neat.LogLevelInfo
	require.NoError(t, err)

	// Check if output dir exists
	err = utils.CreateOutputDir(outDirPath)
	require.NoError(t, err, "Failed to create output directory")

	// Running POLE2 Markov parallel experiment
	opts.NumRuns = 5
	exp := experiment.Experiment{
		Id:     0,
		Trials: make(experiment.Trials, opts.NumRuns),
	}
	evaluator := NewCartDoublePoleParallelGenerationEvaluator(outDirPath, true, ContinuousAction, 50)
	err = exp.Execute(opts.NeatContext(), startGenome, evaluator, nil)
	require.NoError(t, err, "Failed to perform POLE2 Markov parallel experiment")

	// Find winner statistics
	avgNodes, avgGenes, avgEvals, _ := exp.AvgWinnerStatistics()

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
	for _, t := range exp.Trials {
		meanComplexity += t.ChampionsComplexities().Mean()
		meanDiversity += t.Diversity().Mean()
		meanAge += t.ChampionSpeciesAges().Mean()
	}
	count := float64(len(exp.Trials))
	meanComplexity /= count
	meanDiversity /= count
	meanAge /= count
	t.Logf("Mean best organisms: complexity=%.1f, diversity=%.1f, age=%.1f\n", meanComplexity, meanDiversity, meanAge)

	solvedTrials := 0
	for _, tr := range exp.Trials {
		if tr.Solved() {
			solvedTrials++
		}
	}

	t.Logf("Trials solved/run: %d/%d", solvedTrials, len(exp.Trials))

	assert.NotZero(t, solvedTrials, "Failed to solve at least one trial. Need to be checked what was going wrong")
}
