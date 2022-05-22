package experiment

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v3/neat/genetics"
	"testing"
	"time"
)

func TestExperiment_Write_Read(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test Encode Decode", Trials: make(Trials, 3)}
	for i := 0; i < len(ex.Trials); i++ {
		ex.Trials[i] = *buildTestTrial(i+1, 10)
	}

	// Write experiment
	var buff bytes.Buffer
	err := ex.Write(&buff)
	require.NoError(t, err, "Failed to write experiment")

	// Read experiment
	data := buff.Bytes()
	newEx := Experiment{}
	err = newEx.Read(bytes.NewBuffer(data))
	require.NoError(t, err, "failed to read experiment")

	// Deep compare results
	assert.Equal(t, ex.Id, newEx.Id)
	assert.Equal(t, ex.Name, newEx.Name)
	require.Len(t, newEx.Trials, len(ex.Trials))

	for i := 0; i < len(ex.Trials); i++ {
		assert.EqualValues(t, ex.Trials[i], newEx.Trials[i])
	}
}

func TestExperiment_Write_writeError(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test Encode Decode", Trials: make(Trials, 3)}
	for i := 0; i < len(ex.Trials); i++ {
		ex.Trials[i] = *buildTestTrial(i+1, 10)
	}

	errWriter := ErrorWriter(1)
	err := ex.Write(&errWriter)
	assert.EqualError(t, err, alwaysErrorText)
}

func TestExperiment_Read_readError(t *testing.T) {
	errReader := ErrorReader(1)

	newEx := Experiment{}
	err := newEx.Read(&errReader)
	assert.EqualError(t, err, alwaysErrorText)
}

func TestExperiment_WriteNPZ(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test Encode Decode", Trials: make(Trials, 3)}
	for i := 0; i < len(ex.Trials); i++ {
		ex.Trials[i] = *buildTestTrial(i+1, 10)
	}

	// Write experiment
	var buff bytes.Buffer
	err := ex.Write(&buff)
	require.NoError(t, err, "Failed to write experiment")
	assert.True(t, buff.Len() > 0)
}

func TestExperiment_WriteNPZ_writeError(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test Encode Decode", Trials: make(Trials, 3)}
	for i := 0; i < len(ex.Trials); i++ {
		ex.Trials[i] = *buildTestTrial(i+1, 10)
	}

	errWriter := ErrorWriter(1)
	err := ex.Write(&errWriter)
	assert.EqualError(t, err, alwaysErrorText)
}

func TestExperiment_AvgTrialDuration(t *testing.T) {
	trials := Trials{
		Trial{Duration: time.Duration(3)},
		Trial{Duration: time.Duration(10)},
		Trial{Duration: time.Duration(2)},
	}
	ex := Experiment{Id: 1, Name: "Test AvgTrialDuration", Trials: trials}
	duration := ex.AvgTrialDuration()
	assert.Equal(t, time.Duration(5), duration)
}

func TestExperiment_AvgTrialDuration_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test AvgTrialDuration_emptyTrials", Trials: Trials{}}
	duration := ex.AvgTrialDuration()
	assert.Equal(t, EmptyDuration, duration)
}

func TestExperiment_AvgEpochDuration(t *testing.T) {
	durations := [][]time.Duration{
		{time.Duration(3), time.Duration(10), time.Duration(2)},
		{time.Duration(1), time.Duration(1), time.Duration(1)},
	}
	trials := Trials{
		*buildTestTrialWithGenerationsDuration(durations[0]),
		*buildTestTrialWithGenerationsDuration(durations[1]),
	}
	ex := Experiment{Id: 1, Name: "Test AvgEpochDuration", Trials: trials}
	duration := ex.AvgEpochDuration()
	assert.Equal(t, time.Duration(3), duration)
}

func TestExperiment_AvgEpochDuration_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test AvgEpochDuration_emptyTrials", Trials: Trials{}}
	duration := ex.AvgEpochDuration()
	assert.Equal(t, EmptyDuration, duration)
}

func TestExperiment_AvgGenerationsPerTrial(t *testing.T) {
	numGenerations := []int{5, 8, 6, 1}
	trials := Trials{
		*buildTestTrial(0, numGenerations[0]),
		*buildTestTrial(1, numGenerations[1]),
		*buildTestTrial(2, numGenerations[2]),
		*buildTestTrial(3, numGenerations[3]),
	}
	ex := Experiment{Id: 1, Name: "Test AvgGenerationsPerTrial", Trials: trials}
	gens := ex.AvgGenerationsPerTrial()
	assert.Equal(t, 5.0, gens)
}

func TestExperiment_AvgGenerationsPerTrial_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test AvgGenerationsPerTrial_emptyTrials", Trials: Trials{}}
	gens := ex.AvgGenerationsPerTrial()
	assert.Equal(t, 0.0, gens)
}

func TestExperiment_MostRecentTrialEvalTime(t *testing.T) {
	now := time.Now()
	trials := Trials{
		Trial{
			Generations: Generations{Generation{Executed: now}},
		},
		Trial{
			Generations: Generations{Generation{Executed: now.Add(time.Duration(-1))}},
		},
		Trial{
			Generations: Generations{Generation{Executed: now.Add(time.Duration(-2))}},
		},
	}
	ex := Experiment{Id: 1, Name: "Test MostRecentTrialEvalTime", Trials: trials}
	mostRecent := ex.MostRecentTrialEvalTime()
	assert.Equal(t, now, mostRecent)
}

func TestExperiment_MostRecentTrialEvalTime_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test MostRecentTrialEvalTime_emptyTrials", Trials: Trials{}}
	mostRecent := ex.MostRecentTrialEvalTime()
	assert.Equal(t, time.Time{}, mostRecent)
}

func TestExperiment_BestOrganism(t *testing.T) {
	fitnessMultipliers := Floats{1.0, 2.0, 3.0}
	trials := make(Trials, len(fitnessMultipliers))
	for i, fm := range fitnessMultipliers {
		trials[i] = *buildTestTrialWithFitnessMultiplier(i, i+2, fm)
	}
	ex := Experiment{Id: 1, Name: "Test BestOrganism", Trials: trials}
	bestOrg, trialId, ok := ex.BestOrganism(true)
	assert.True(t, ok)
	// the last trial
	assert.Equal(t, 2, trialId)
	// the best organism of last generation of last trial
	assert.Equal(t, fitnessScore(2+2)*fitnessMultipliers[2], bestOrg.Fitness)
}

func TestExperiment_BestOrganism_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test BestOrganism_emptyTrials", Trials: Trials{}}
	bestOrg, trialId, ok := ex.BestOrganism(true)
	assert.False(t, ok)
	assert.Equal(t, -1, trialId)
	assert.Nil(t, bestOrg)
}

func TestExperiment_Solved(t *testing.T) {
	trials := Trials{
		*buildTestTrial(1, 2),
		*buildTestTrial(2, 3),
		*buildTestTrial(3, 5),
	}
	ex := Experiment{Id: 1, Name: "Test Solved", Trials: trials}
	solved := ex.Solved()
	assert.True(t, solved)
}

func TestExperiment_Solved_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test Solved_emptyTrials", Trials: Trials{}}
	solved := ex.Solved()
	assert.False(t, solved)
}

func TestExperiment_BestFitness(t *testing.T) {
	fitnessMultipliers := Floats{1.0, 2.0, 3.0}
	trials := make(Trials, len(fitnessMultipliers))
	expectedFitness := make(Floats, len(fitnessMultipliers))
	for i, fm := range fitnessMultipliers {
		trials[i] = *buildTestTrialWithFitnessMultiplier(i, i+2, fm)
		expectedFitness[i] = fitnessScore(i+2) * fm
	}
	ex := Experiment{Id: 1, Name: "Test ChampionFitness", Trials: trials}
	bestFitness := ex.BestFitness()
	assert.Equal(t, len(expectedFitness), len(bestFitness))
	assert.EqualValues(t, expectedFitness, bestFitness)
}

func TestExperiment_BestFitness_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test BestFitness_emptyTrials", Trials: Trials{}}
	bestFitness := ex.BestFitness()
	assert.Equal(t, 0, len(bestFitness))
}

func TestExperiment_BestSpeciesAge(t *testing.T) {
	trials := Trials{
		*buildTestTrial(10, 1),
		*buildTestTrial(20, 2),
		*buildTestTrial(30, 3),
	}
	// assign species to the best organisms
	expected := Floats{10, 15, 1}
	for i, t := range trials {
		if org, ok := t.BestOrganism(false); ok {
			org.Species = genetics.NewSpecies(i)
			org.Species.Age = int(expected[i])
		}
	}

	ex := Experiment{Id: 1, Name: "Test BestSpeciesAge", Trials: trials}
	bestAge := ex.BestSpeciesAge()
	assert.EqualValues(t, expected, bestAge)
}

func TestExperiment_BestSpeciesAge_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test BestSpeciesAge_emptyTrials", Trials: Trials{}}
	bestAge := ex.BestSpeciesAge()
	assert.Equal(t, 0, len(bestAge))
}

func TestExperiment_BestComplexity(t *testing.T) {
	trials := Trials{
		*buildTestTrialWithBestOrganismGenesis(1, 3),
		*buildTestTrialWithBestOrganismGenesis(2, 4),
		*buildTestTrialWithBestOrganismGenesis(3, 2),
	}
	ex := Experiment{Id: 1, Name: "Test BestComplexity", Trials: trials}
	bestComplexity := ex.BestComplexity()
	expected := Floats{7.0, 7.0, 7.0}
	assert.EqualValues(t, expected, bestComplexity)
}

func TestExperiment_BestComplexity_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test BestComplexity_emptyTrials", Trials: Trials{}}
	bestComplexity := ex.BestComplexity()
	assert.Equal(t, 0, len(bestComplexity))
}

func TestExperiment_AvgDiversity(t *testing.T) {
	trials := Trials{
		*buildTestTrial(1, 2),
		*buildTestTrial(1, 3),
		*buildTestTrial(1, 5),
	}
	ex := Experiment{Id: 1, Name: "Test AvgDiversity", Trials: trials}

	avgDiversity := ex.AvgDiversity()
	expected := Floats{testDiversity, testDiversity, testDiversity}
	assert.EqualValues(t, expected, avgDiversity)
}

func TestExperiment_AvgDiversity_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test AvgDiversity_emptyTrials", Trials: Trials{}}
	avgDiversity := ex.AvgDiversity()
	assert.Equal(t, 0, len(avgDiversity))
}

func TestExperiment_EpochsPerTrial(t *testing.T) {
	expected := Floats{2, 3, 5}
	trials := Trials{
		*buildTestTrial(1, int(expected[0])),
		*buildTestTrial(1, int(expected[1])),
		*buildTestTrial(1, int(expected[2])),
	}
	ex := Experiment{Id: 1, Name: "Test EpochsPerTrial", Trials: trials}

	epochs := ex.EpochsPerTrial()
	assert.EqualValues(t, expected, epochs)
}

func TestExperiment_EpochsPerTrial_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test EpochsPerTrial_emptyTrials", Trials: Trials{}}
	epochs := ex.EpochsPerTrial()
	assert.Equal(t, 0, len(epochs))
}

func TestExperiment_TrialsSolved(t *testing.T) {
	solvedExpected := 2
	trials := createTrialsWithNSolved([]int{2, 3, 5}, solvedExpected)

	ex := Experiment{Id: 1, Name: "Test TrialsSolved", Trials: trials}
	solved := ex.TrialsSolved()
	assert.Equal(t, solvedExpected, solved)
}

func TestExperiment_TrialsSolved_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test TrialsSolved_emptyTrials", Trials: Trials{}}
	solved := ex.TrialsSolved()
	assert.Equal(t, 0, solved)
}

func TestExperiment_SuccessRate(t *testing.T) {
	solvedExpected := 2
	trials := createTrialsWithNSolved([]int{2, 3, 5}, solvedExpected)

	ex := Experiment{Id: 1, Name: "Test SuccessRate", Trials: trials}
	successRate := ex.SuccessRate()
	expectedRate := float64(solvedExpected) / 3.0
	assert.Equal(t, expectedRate, successRate)
}

func TestExperiment_SuccessRate_emptyTrials(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test TrialsSolved_emptyTrials", Trials: Trials{}}
	successRate := ex.SuccessRate()
	assert.Equal(t, 0.0, successRate)
}

func createTrialsWithNSolved(generations []int, solvedNumber int) Trials {
	trials := make(Trials, len(generations))
	for i := range generations {
		trials[i] = *buildTestTrial(i, generations[i])
	}

	for _, trial := range trials {
		solved := solvedNumber > 0
		solvedNumber -= 1
		for j := range trial.Generations {
			trial.Generations[j].Solved = solved
		}
	}
	return trials
}
