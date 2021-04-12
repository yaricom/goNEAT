package neat

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"os"
	"testing"
)

func TestLoadContext(t *testing.T) {
	config, err := os.Open("../data/xor_test.neat")
	require.NoError(t, err)

	// Load Neat Context
	nc := LoadContext(config)
	checkNeatContext(nc, t)
}

func TestNeatContext_LoadContext(t *testing.T) {
	config, err := os.Open("../data/xor_test.neat.yml")
	require.NoError(t, err)

	// Load YAML context
	nc := NewNeatContext()
	err = nc.LoadContext(config)
	require.NoError(t, err, "failed to load context")

	checkNeatContext(nc, t)

	// check activators
	require.Len(t, nc.NodeActivators, 4, "wrong node activators size")
	activators := []math.NodeActivationType{math.SigmoidBipolarActivation,
		math.GaussianBipolarActivation, math.LinearAbsActivation, math.SineActivation}
	probs := []float64{0.25, 0.35, 0.15, 0.25}
	for i, a := range activators {
		assert.Equal(t, a, nc.NodeActivators[i], "wrong node activator type at: %d", i)
		assert.Equal(t, probs[i], nc.NodeActivatorsProb[i], "wrong probability at: %d", i)

	}
}

func checkNeatContext(nc *NeatContext, t *testing.T) {
	assert.Equal(t, 0.5, nc.TraitParamMutProb)
	assert.Equal(t, 1.0, nc.TraitMutationPower)
	assert.Equal(t, 2.5, nc.WeightMutPower)
	assert.Equal(t, 1.0, nc.DisjointCoeff)
	assert.Equal(t, 1.0, nc.ExcessCoeff)
	assert.Equal(t, 0.4, nc.MutdiffCoeff)
	assert.Equal(t, 3.0, nc.CompatThreshold)
	assert.Equal(t, 1.0, nc.AgeSignificance)
	assert.Equal(t, 0.2, nc.SurvivalThresh)
	assert.Equal(t, 0.25, nc.MutateOnlyProb)
	assert.Equal(t, 0.1, nc.MutateRandomTraitProb)
	assert.Equal(t, 0.1, nc.MutateLinkTraitProb)
	assert.Equal(t, 0.1, nc.MutateNodeTraitProb)
	assert.Equal(t, 0.9, nc.MutateLinkWeightsProb)
	assert.Equal(t, 0.0, nc.MutateToggleEnableProb)
	assert.Equal(t, 0.0, nc.MutateGeneReenableProb)
	assert.Equal(t, 0.03, nc.MutateAddNodeProb)
	assert.Equal(t, 0.08, nc.MutateAddLinkProb)
	assert.Equal(t, 0.5, nc.MutateConnectSensors)
	assert.Equal(t, 0.001, nc.InterspeciesMateRate)
	assert.Equal(t, 0.3, nc.MateMultipointProb)
	assert.Equal(t, 0.3, nc.MateMultipointAvgProb)
	assert.Equal(t, 0.3, nc.MateSinglepointProb)
	assert.Equal(t, 0.2, nc.MateOnlyProb)
	assert.Equal(t, 0.0, nc.RecurOnlyProb)
	assert.Equal(t, 200, nc.PopSize)
	assert.Equal(t, 50, nc.DropOffAge)
	assert.Equal(t, 50, nc.NewLinkTries)
	assert.Equal(t, 10, nc.PrintEvery)
	assert.Equal(t, 0, nc.BabiesStolen)
	assert.Equal(t, 100, nc.NumRuns)
	assert.Equal(t, 100, nc.NumGenerations)
	assert.Equal(t, 0, nc.EpochExecutorType)
	assert.Equal(t, 1, nc.GenCompatMethod)
}
