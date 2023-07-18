package experiment

import (
	"context"
	"errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"testing"
)

const (
	xorGenomePath = "../data/xorstartgenes.yml"
	xorConfigPath = "../data/xor_test.neat.yml"
)

type MockedGenerationEvaluator struct {
	mock.Mock
}

func (m *MockedGenerationEvaluator) GenerationEvaluate(ctx context.Context, pop *genetics.Population, epoch *Generation) error {
	args := m.Called(ctx, pop, epoch)
	return args.Error(0)
}

type MockedTrialRunObserver struct {
	mock.Mock
}

func (m *MockedTrialRunObserver) TrialRunStarted(trial *Trial) {
	m.Called(trial)
}

func (m *MockedTrialRunObserver) TrialRunFinished(trial *Trial) {
	m.Called(trial)
}

func (m *MockedTrialRunObserver) EpochEvaluated(trial *Trial, epoch *Generation) {
	m.Called(trial, epoch)
}

func readTestGenome() (*genetics.Genome, error) {
	r, err := genetics.NewGenomeReaderFromFile(xorGenomePath)
	if err != nil {
		return nil, err
	}
	return r.Read()
}

func TestExperiment_Execute_no_NEAT_options(t *testing.T) {
	exp := Experiment{
		Id: 0,
	}
	genome, err := readTestGenome()
	require.NoError(t, err)
	err = exp.Execute(context.Background(), genome, &MockedGenerationEvaluator{}, &MockedTrialRunObserver{})
	assert.Error(t, err, neat.ErrNEATOptionsNotFound.Error())
}

func TestExperiment_Execute(t *testing.T) {
	exp := Experiment{
		Id: 0,
	}
	genome, err := readTestGenome()
	require.NoError(t, err, "failed to read XOR genome")
	opts, err := neat.ReadNeatOptionsFromFile(xorConfigPath)
	require.NoError(t, err, "failed to read NEAT options")
	opts.NumRuns = 10
	opts.NumGenerations = 10
	ctx := neat.NewContext(context.Background(), opts)

	genEvaluator := &MockedGenerationEvaluator{}
	trialsObserver := &MockedTrialRunObserver{}

	// setup expectations
	genEvaluatorCallsNum := opts.NumRuns * opts.NumGenerations
	genEvaluator.On("GenerationEvaluate", ctx, mock.Anything, mock.Anything).Return(nil)

	trialsNum := opts.NumRuns
	trialsObserver.On("TrialRunStarted", mock.Anything).Return(nil)
	trialsObserver.On("TrialRunFinished", mock.Anything).Return(nil)
	trialsObserver.On("EpochEvaluated", mock.Anything, mock.Anything).Return(nil)

	err = exp.Execute(ctx, genome, genEvaluator, trialsObserver)
	require.NoError(t, err, "failed to execute experiment")
	assert.Equal(t, trialsNum, len(exp.Trials), "wrong number of trials collected")
	assert.True(t, exp.AvgTrialDuration() > 0)
	assert.True(t, exp.AvgEpochDuration() > 0)
	assert.EqualValues(t, opts.NumGenerations, exp.AvgGenerationsPerTrial())
	assert.False(t, exp.Solved())

	// check mocks assertions
	genEvaluator.AssertNumberOfCalls(t, "GenerationEvaluate", genEvaluatorCallsNum)
	trialsObserver.AssertNumberOfCalls(t, "TrialRunStarted", trialsNum)
	trialsObserver.AssertNumberOfCalls(t, "TrialRunFinished", trialsNum)
	trialsObserver.AssertNumberOfCalls(t, "TrialRunFinished", trialsNum)

	// assert that the expectations were met
	genEvaluator.AssertExpectations(t)
}

func TestExperiment_Execute_evaluation_error(t *testing.T) {
	exp := Experiment{
		Id: 0,
	}
	genome, err := readTestGenome()
	require.NoError(t, err, "failed to read XOR genome")
	opts, err := neat.ReadNeatOptionsFromFile(xorConfigPath)
	require.NoError(t, err, "failed to read NEAT options")
	opts.NumRuns = 10
	opts.NumGenerations = 10
	ctx := neat.NewContext(context.Background(), opts)

	genEvaluator := &MockedGenerationEvaluator{}

	// setup expectations
	evaluationError := errors.New("evaluation error")
	genEvaluator.On("GenerationEvaluate", ctx, mock.Anything, mock.Anything).Return(evaluationError)

	err = exp.Execute(ctx, genome, genEvaluator, nil)
	require.Error(t, err, evaluationError.Error())

	// assert that the expectations were met
	genEvaluator.AssertNumberOfCalls(t, "GenerationEvaluate", 1)
	genEvaluator.AssertExpectations(t)
}
