package neat

import (
	"errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v3/neat/math"
	"os"
	"testing"
)

const (
	alwaysErrorText     = "always be failing"
	xorOptionsFilePlain = "../data/xor_test.neat"
	xorOptionsFileYaml  = "../data/xor_test.neat.yml"
)

var errFoo = errors.New(alwaysErrorText)

type ErrorReader int

func (e ErrorReader) Read(_ []byte) (n int, err error) {
	return 0, errFoo
}

func TestLoadNeatOptions(t *testing.T) {
	config, err := os.Open(xorOptionsFilePlain)
	require.NoError(t, err)

	// Load Neat Context
	opts, err := LoadNeatOptions(config)
	require.NoError(t, err)
	checkNeatOptions(opts, t)
}

func TestLoadNeatOptions_readError(t *testing.T) {
	errorReader := ErrorReader(1)
	opts, err := LoadNeatOptions(&errorReader)
	assert.EqualError(t, err, alwaysErrorText)
	assert.Nil(t, opts)
}

func TestLoadYAMLOptions(t *testing.T) {
	config, err := os.Open(xorOptionsFileYaml)
	require.NoError(t, err)

	// Load YAML context
	opts, err := LoadYAMLOptions(config)
	require.NoError(t, err, "failed to load options")

	checkNeatOptions(opts, t)

	// check activators
	require.Len(t, opts.NodeActivators, 4, "wrong node activators size")
	activators := []math.NodeActivationType{math.SigmoidBipolarActivation,
		math.GaussianBipolarActivation, math.LinearAbsActivation, math.SineActivation}
	probs := []float64{0.25, 0.35, 0.15, 0.25}
	for i, a := range activators {
		assert.Equal(t, a, opts.NodeActivators[i], "wrong node activator type at: %d", i)
		assert.Equal(t, probs[i], opts.NodeActivatorsProb[i], "wrong probability at: %d", i)

	}
}

func TestLoadYAMLOptions_readError(t *testing.T) {
	errorReader := ErrorReader(1)
	opts, err := LoadYAMLOptions(&errorReader)
	assert.EqualError(t, err, alwaysErrorText)
	assert.Nil(t, opts)
}

func TestOptions_NeatContext(t *testing.T) {
	config, err := os.Open(xorOptionsFileYaml)
	require.NoError(t, err)

	// Load YAML context
	opts, err := LoadYAMLOptions(config)
	require.NoError(t, err, "failed to load options")

	// check that NEAT context has options inside
	ctx := opts.NeatContext()

	nOpts, ok := FromContext(ctx)
	require.True(t, ok, "options not found")
	assert.NotNil(t, nOpts)
}

func TestReadNeatOptionsFromFile(t *testing.T) {
	opts, err := ReadNeatOptionsFromFile(xorOptionsFilePlain)
	require.NoError(t, err, "failed to read NEAT options with PLAIN encoding")
	assert.NotNil(t, opts)

	opts, err = ReadNeatOptionsFromFile(xorOptionsFileYaml)
	require.NoError(t, err, "failed to read NEAT options with YAML encoding")
	assert.NotNil(t, opts)
}

func TestReadNeatOptionsFromFile_error(t *testing.T) {
	opts, err := ReadNeatOptionsFromFile("file doesnt exist")
	assert.Error(t, err)
	assert.Nil(t, opts)
}

func checkNeatOptions(nc *Options, t *testing.T) {
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
	assert.Equal(t, EpochExecutorTypeSequential, nc.EpochExecutorType)
	assert.Equal(t, GenomeCompatibilityMethodFast, nc.GenCompatMethod)
}
