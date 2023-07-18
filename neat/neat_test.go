package neat

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/neat/math"
	"testing"
)

func TestOptions_NeatContext(t *testing.T) {
	opts := &Options{
		CompatThreshold:    0.5,
		PopSize:            10,
		NodeActivators:     []math.NodeActivationType{math.GaussianBipolarActivation},
		NodeActivatorsProb: []float64{1.0},
	}

	// check that NEAT context has options inside
	ctx := opts.NeatContext()

	nOpts, ok := FromContext(ctx)
	require.True(t, ok, "options not found")
	assert.NotNil(t, nOpts)
	assert.EqualValues(t, opts, nOpts)
}

func TestOptions_RandomNodeActivationType_noActivators(t *testing.T) {
	opts := &Options{
		CompatThreshold: 0.5,
		PopSize:         10,
	}
	activator, err := opts.RandomNodeActivationType()
	assert.EqualError(t, err, ErrNoActivatorsRegistered.Error())
	assert.EqualValues(t, 0, activator)
}

func TestOptions_RandomNodeActivationType_activatorsProbabilitiesNumberMismatch(t *testing.T) {
	opts := &Options{
		NodeActivators:     []math.NodeActivationType{math.SigmoidApproximationActivation, math.SigmoidBipolarActivation},
		NodeActivatorsProb: []float64{0.5},
	}
	activator, err := opts.RandomNodeActivationType()
	assert.EqualError(t, err, ErrActivatorsProbabilitiesNumberMismatch.Error())
	assert.EqualValues(t, 0, activator)
}

func TestOptions_RandomNodeActivationType_singleValue(t *testing.T) {
	opts := &Options{
		NodeActivators:     []math.NodeActivationType{math.GaussianBipolarActivation},
		NodeActivatorsProb: []float64{1.0},
	}
	activator, err := opts.RandomNodeActivationType()
	require.NoError(t, err)
	assert.Equal(t, math.GaussianBipolarActivation, activator)
}

func TestOptions_RandomNodeActivationType(t *testing.T) {
	opts := &Options{
		NodeActivators:     []math.NodeActivationType{math.SigmoidApproximationActivation, math.SigmoidBipolarActivation},
		NodeActivatorsProb: []float64{0.5, 0.5},
	}
	activator, err := opts.RandomNodeActivationType()
	require.NoError(t, err)
	res := activator == math.SigmoidApproximationActivation || activator == math.SigmoidBipolarActivation
	assert.True(t, res)
}
