package experiment

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestTrial_AvgEpochDuration(t *testing.T) {
	durations := []time.Duration{time.Duration(3), time.Duration(10), time.Duration(2)}
	trial := buildTestTrialWithGenerationsDuration(durations)
	avg := trial.AvgEpochDuration()
	assert.Equal(t, time.Duration(5), avg)
}

func TestTrial_AvgEpochDuration_emptyEpochs(t *testing.T) {
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	avg := trial.AvgEpochDuration()
	assert.Equal(t, EmptyDuration, avg)
}

func TestTrial_RecentEpochEvalTime(t *testing.T) {
	now := time.Now().Add(-10 * time.Second)
	trial := buildTestTrial(1, 3)
	evalTime := trial.RecentEpochEvalTime()
	assert.True(t, evalTime.After(now))
}

func TestTrial_RecentEpochEvalTime_emptyEpochs(t *testing.T) {
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	ex := trial.RecentEpochEvalTime()
	assert.Equal(t, time.Time{}, ex)
}

func TestTrial_BestOrganism(t *testing.T) {
	trial := buildTestTrial(1, 3)
	org, ok := trial.BestOrganism(true)
	assert.True(t, ok)
	assert.NotNil(t, org)
	fit := fitnessScore(3)
	assert.Equal(t, fit, org.Fitness)
}

func TestTrial_BestOrganism_emptyEpochs(t *testing.T) {
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	org, ok := trial.BestOrganism(true)
	assert.False(t, ok)
	assert.Nil(t, org)
}

func TestTrial_Solved(t *testing.T) {
	trial := buildTestTrial(1, 5)
	solved := trial.Solved()
	assert.True(t, solved)
}

func TestTrial_Solved_emptyEpochs(t *testing.T) {
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	solved := trial.Solved()
	assert.False(t, solved)
}

func TestTrial_ChampionFitness(t *testing.T) {
	numGen := 4
	trial := buildTestTrial(1, numGen)
	fitness := trial.ChampionsFitness()
	assert.Equal(t, numGen, len(fitness))
	expected := make(Floats, numGen)
	for i := 0; i < numGen; i++ {
		expected[i] = fitnessScore(i + 1)
	}
	assert.EqualValues(t, expected, fitness)
}

func TestTrial_ChampionFitness_emptyEpochs(t *testing.T) {
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	fitness := trial.ChampionsFitness()
	assert.Equal(t, 0, len(fitness))
}

func TestTrial_ChampionSpeciesAge(t *testing.T) {
	numGen := 4
	trial := buildTestTrial(1, numGen)

	// assign species to the best organisms
	expected := Floats{10, 15, 1, 5}
	for i, g := range trial.Generations {
		g.Champion.Species = genetics.NewSpecies(i)
		g.Champion.Species.Age = int(expected[i])
	}
	age := trial.ChampionSpeciesAges()
	assert.Equal(t, numGen, len(age))
}

func TestTrial_ChampionSpeciesAge_emptyEpochs(t *testing.T) {
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	age := trial.ChampionSpeciesAges()
	assert.Equal(t, 0, len(age))
}

func TestTrial_ChampionComplexity(t *testing.T) {
	numGen := 4
	trial := buildTestTrialWithBestOrganismGenesis(1, numGen)

	compl := trial.ChampionsComplexities()
	assert.Equal(t, numGen, len(compl))
	expected := make(Floats, numGen)
	for i := 0; i < numGen; i++ {
		expected[i] = 7
	}
	assert.EqualValues(t, expected, compl)
}

func TestTrial_ChampionComplexity_emptyEpochs(t *testing.T) {
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	compl := trial.ChampionsComplexities()
	assert.Equal(t, 0, len(compl))
}

func TestTrial_Diversity(t *testing.T) {
	numGen := 4
	trial := buildTestTrial(1, numGen)
	div := trial.Diversity()
	assert.Equal(t, numGen, len(div))
	expected := make(Floats, numGen)
	for i := 0; i < numGen; i++ {
		expected[i] = testDiversity
	}
	assert.EqualValues(t, expected, div)
}

func TestTrial_Diversity_emptyEpochs(t *testing.T) {
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	div := trial.Diversity()
	assert.Equal(t, 0, len(div))
}

func TestTrial_Average(t *testing.T) {
	numGen := 4
	trial := buildTestTrial(1, numGen)
	fitness, age, complexity := trial.Average()
	assert.Equal(t, numGen, len(fitness))
	assert.Equal(t, numGen, len(age))
	assert.Equal(t, numGen, len(complexity))
	expectedAge := make(Floats, numGen)
	expectedComplexity := make(Floats, numGen)
	expectedFitness := make(Floats, numGen)
	for i := 0; i < numGen; i++ {
		expectedAge[i] = testAge.Mean()
		expectedComplexity[i] = testComplexity.Mean()
		fit := fitnessScore(i + 1)
		fitSlice := append(testFitness, fit)
		expectedFitness[i] = fitSlice.Mean()
	}
	assert.EqualValues(t, expectedAge, age)
	assert.EqualValues(t, expectedFitness, fitness)
	assert.EqualValues(t, expectedComplexity, complexity)
}

func TestTrial_Average_emptyEpochs(t *testing.T) {
	numGen := 0
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	fitness, age, complexity := trial.Average()
	assert.Equal(t, numGen, len(fitness))
	assert.Equal(t, numGen, len(age))
	assert.Equal(t, numGen, len(complexity))
}

func TestTrial_WinnerStatistics(t *testing.T) {
	numGen := 4
	trial := buildTestTrial(1, numGen)
	nodes, genes, evals, diversity := trial.WinnerStatistics()
	assert.Equal(t, testWinnerNodes, nodes)
	assert.Equal(t, testWinnerGenes, genes)
	assert.Equal(t, testWinnerEvals, evals)
	assert.Equal(t, testDiversity, diversity)
	assert.NotNil(t, trial.WinnerGeneration)
}

func TestTrial_WinnerStatistics_emptyEpochs(t *testing.T) {
	trial := Trial{Id: 1, Generations: make([]Generation, 0)}
	nodes, genes, evals, diversity := trial.WinnerStatistics()
	assert.Equal(t, -1, nodes)
	assert.Equal(t, -1, genes)
	assert.Equal(t, -1, evals)
	assert.Equal(t, -1, diversity)
	assert.Nil(t, trial.WinnerGeneration)
}

func TestTrial_Encode_Decode(t *testing.T) {
	trial := buildTestTrial(1, 3)

	var buff bytes.Buffer
	enc := gob.NewEncoder(&buff)

	// encode trial
	err := trial.Encode(enc)
	require.NoError(t, err, "failed to encode Trial")

	// decode trial
	data := buff.Bytes()
	dec := gob.NewDecoder(bytes.NewBuffer(data))

	decTrial := Trial{}
	err = decTrial.Decode(dec)
	require.NoError(t, err, "failed to decode trial")

	// do deep compare of Trail fields
	assert.EqualValues(t, *trial, decTrial)
}

func buildTestTrial(id, numGenerations int) *Trial {
	return buildTestTrialWithFitnessMultiplier(id, numGenerations, 1.0)
}

func buildTestTrialWithFitnessMultiplier(id, numGenerations int, fitnessMultiplier float64) *Trial {
	trial := Trial{Id: id, Generations: make([]Generation, numGenerations)}
	for i := 0; i < numGenerations; i++ {
		trial.Generations[i] = *buildTestGeneration(i+1, fitnessScore(i+1)*fitnessMultiplier)
	}
	return &trial
}

func buildTestTrialWithGenerationsDuration(durations []time.Duration) *Trial {
	return &Trial{Id: rand.Int(), Generations: buildTestGenerationsWithDuration(durations)}
}

func buildTestTrialWithBestOrganismGenesis(id, numGenerations int) *Trial {
	trial := buildTestTrial(id, numGenerations)
	// do genesis of best organisms
	for i := range trial.Generations {
		if err := trial.Generations[i].Champion.UpdatePhenotype(); err != nil {
			neat.WarnLog(fmt.Sprintf("Failed to crete phenotype, reason: %s", err))
		}
	}
	return trial
}

func fitnessScore(index int) float64 {
	return float64(index) * math.E
}
