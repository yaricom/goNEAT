package genetics

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v3/neat"
	"github.com/yaricom/goNEAT/v3/neat/network"
	"testing"
)

func TestGenome_Compatibility_Linear(t *testing.T) {
	//rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)

	// Configuration
	conf := neat.Options{
		DisjointCoeff:   0.5,
		ExcessCoeff:     0.5,
		MutdiffCoeff:    0.5,
		GenCompatMethod: neat.GenomeCompatibilityMethodLinear,
	}

	// Test fully compatible
	comp := gnome1.compatibility(gnome2, &conf)
	assert.Equal(t, 0.0, comp, "not fully compatible")

	// Test incompatible
	gnome2.Genes = append(gnome2.Genes, NewGene(1.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 10, 1.0))
	comp = gnome1.compatibility(gnome2, &conf)
	assert.Equal(t, 0.5, comp)

	gnome2.Genes = append(gnome2.Genes, NewGene(2.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 5, 1.0))
	comp = gnome1.compatibility(gnome2, &conf)
	assert.Equal(t, 1.0, comp)

	gnome2.Genes[1].MutationNum = 6.0
	comp = gnome1.compatibility(gnome2, &conf)
	assert.Equal(t, 2.0, comp)
}

func TestGenome_Compatibility_Fast(t *testing.T) {
	//rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)

	// Configuration
	conf := neat.Options{
		DisjointCoeff:   0.5,
		ExcessCoeff:     0.5,
		MutdiffCoeff:    0.5,
		GenCompatMethod: neat.GenomeCompatibilityMethodFast,
	}

	// Test fully compatible
	comp := gnome1.compatibility(gnome2, &conf)
	assert.Equal(t, 0.0, comp, "not fully compatible")

	// Test incompatible
	gnome2.Genes = append(gnome2.Genes, NewGene(1.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 10, 1.0))
	comp = gnome1.compatibility(gnome2, &conf)
	assert.Equal(t, 0.5, comp)

	gnome2.Genes = append(gnome2.Genes, NewGene(2.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 5, 1.0))
	comp = gnome1.compatibility(gnome2, &conf)
	assert.Equal(t, 1.0, comp)

	gnome2.Genes[1].MutationNum = 6.0
	comp = gnome1.compatibility(gnome2, &conf)
	assert.Equal(t, 2.0, comp)
}

func TestGenome_Compatibility_Duplicate(t *testing.T) {
	//rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2, err := gnome1.duplicate(2)
	require.NoError(t, err, "duplication failed")

	// Configuration
	conf := neat.Options{
		DisjointCoeff: 0.5,
		ExcessCoeff:   0.5,
		MutdiffCoeff:  0.5,
	}

	// Test fully compatible
	comp := gnome1.compatibility(gnome2, &conf)
	assert.Equal(t, 0.0, comp, "not fully compatible")
}
