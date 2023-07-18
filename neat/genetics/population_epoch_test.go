package genetics

import (
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/math"
	"math/rand"
	"testing"
)

func sequentialExecutorNextEpoch(pop *Population, opts *neat.Options) error {
	ex := SequentialPopulationEpochExecutor{}

	for i := 0; i < 100; i++ {
		err := ex.NextEpoch(opts.NeatContext(), i+1, pop)
		if err != nil {
			return errors.Wrapf(err, "failed at: %d epoch", i)
		}
	}
	return nil

}

func parallelExecutorNextEpoch(pop *Population, opts *neat.Options) error {
	ex := ParallelPopulationEpochExecutor{}

	for i := 0; i < 100; i++ {
		err := ex.NextEpoch(opts.NeatContext(), i+1, pop)
		if err != nil {
			return errors.Wrapf(err, "failed at: %d epoch", i)
		}
	}
	return nil
}

func TestPopulationEpochExecutor_NextEpoch(t *testing.T) {
	rand.Seed(42)
	in, out, maxHidden, n := 3, 2, 15, 3
	linkProb := 0.8
	conf := &neat.Options{
		CompatThreshold:    0.5,
		DropOffAge:         1,
		PopSize:            30,
		BabiesStolen:       10,
		RecurOnlyProb:      0.2,
		NodeActivators:     []math.NodeActivationType{math.GaussianBipolarActivation},
		NodeActivatorsProb: []float64{1.0},
	}
	neat.LogLevel = neat.LogLevelInfo
	gen, err := newGenomeRand(1, in, out, n, maxHidden, false, linkProb, conf)
	require.NoError(t, err, "failed to create random genome")

	pop, err := NewPopulation(gen, conf)
	require.NoError(t, err, "failed to create population")
	require.NotNil(t, pop, "population expected")

	// test sequential executor
	err = sequentialExecutorNextEpoch(pop, conf)
	assert.NoError(t, err, "failed to run sequential epoch executor")

	// test parallel executor
	err = parallelExecutorNextEpoch(pop, conf)
	assert.NoError(t, err, "failed to run parallel epoch executor")
}
