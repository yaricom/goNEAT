package experiment

import (
	"context"
	"errors"
	"github.com/stretchr/testify/assert"
	"github.com/yaricom/goNEAT/v3/neat"
	"github.com/yaricom/goNEAT/v3/neat/genetics"
	"testing"
)

const alwaysErrorText = "always be failing"

var alwaysError = errors.New(alwaysErrorText)

type ErrorWriter int

func (e ErrorWriter) Write(_ []byte) (int, error) {
	return 0, alwaysError
}

type ErrorReader int

func (e ErrorReader) Read(_ []byte) (n int, err error) {
	return 0, alwaysError
}

func Test_epochExecutorForContext_wrongContext(t *testing.T) {
	ctx := context.Background()
	_, err := epochExecutorForContext(ctx)
	assert.Error(t, err, neat.ErrNEATOptionsNotFound.Error())
}

func Test_epochExecutorForContext_wrongExecutorType(t *testing.T) {
	options := neat.Options{}
	options.EpochExecutorType = "not existing"
	ctx := neat.NewContext(context.Background(), &options)
	evaluator, err := epochExecutorForContext(ctx)
	assert.Error(t, err, "unsupported epoch executor type requested")
	assert.Nil(t, evaluator)
}

func Test_epochExecutorForContext(t *testing.T) {
	testCases := []neat.EpochExecutorType{
		neat.EpochExecutorTypeSequential,
		neat.EpochExecutorTypeParallel,
	}

	for _, tc := range testCases {
		options := neat.Options{}
		options.EpochExecutorType = tc
		ctx := neat.NewContext(context.Background(), &options)
		evaluator, err := epochExecutorForContext(ctx)
		assert.NoError(t, err)
		assert.NotNil(t, evaluator)
		switch tc {
		case neat.EpochExecutorTypeSequential:
			_, ok := evaluator.(*genetics.SequentialPopulationEpochExecutor)
			assert.True(t, ok)
		case neat.EpochExecutorTypeParallel:
			_, ok := evaluator.(*genetics.ParallelPopulationEpochExecutor)
			assert.True(t, ok)
		}
	}
}
