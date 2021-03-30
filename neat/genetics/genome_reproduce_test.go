package genetics

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"math/rand"
	"testing"
)

func TestGenome_mateMultipoint(t *testing.T) {
	rand.Seed(42)
	// Check equal sized gene pools
	//
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)
	genomeId := 3
	fitness1, fitness2 := 1.0, 2.3
	genomeChild, err := gnome1.mateMultipoint(gnome2, genomeId, fitness1, fitness2)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 3, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 4, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")

	// check not equal sized gene pools
	//
	gene := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2],
		gnome1.Nodes[3], false), 4, 0, true)
	gnome1.Genes = append(gnome1.Genes, gene)
	fitness1, fitness2 = 15.0, 2.3
	genomeChild, err = gnome1.mateMultipoint(gnome2, genomeId, fitness1, fitness2)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 3, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 4, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")
}

func TestGenome_mateMultipointModular(t *testing.T) {
	rand.Seed(42)
	// Check equal sized gene pools
	//
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestModularGenome(2)
	genomeId := 3
	fitness1, fitness2 := 1.0, 2.3
	genomeChild, err := gnome1.mateMultipoint(gnome2, genomeId, fitness1, fitness2)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 6, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 7, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")
	assert.Len(t, genomeChild.ControlGenes, 1, "wrong number of control genes")
}

func TestGenome_mateMultipointAvg(t *testing.T) {
	rand.Seed(42)
	// Check equal sized gene pools
	//
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)
	genomeId := 3
	fitness1, fitness2 := 1.0, 2.3
	genomeChild, err := gnome1.mateMultipointAvg(gnome2, genomeId, fitness1, fitness2)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 3, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 4, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")

	// check not equal sized gene pools
	//
	gene := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2],
		gnome1.Nodes[3], false), 4, 0, false)
	gnome1.Genes = append(gnome1.Genes, gene)
	gene2 := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[1],
		gnome1.Nodes[3], true), 4, 0, false)
	gnome2.Genes = append(gnome2.Genes, gene2)

	fitness1, fitness2 = 15.0, 2.3
	genomeChild, err = gnome1.mateMultipointAvg(gnome2, genomeId, fitness1, fitness2)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 4, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 4, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")
}

func TestGenome_mateMultipointAvgModular(t *testing.T) {
	rand.Seed(42)
	// Check equal sized gene pools
	//
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestModularGenome(2)
	genomeId := 3
	fitness1, fitness2 := 1.0, 2.3
	genomeChild, err := gnome1.mateMultipointAvg(gnome2, genomeId, fitness1, fitness2)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 6, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 7, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")
	assert.Len(t, genomeChild.ControlGenes, 1, "wrong number of control genes")
}

func TestGenome_mateSinglePoint(t *testing.T) {
	rand.Seed(42)
	// Check equal sized gene pools
	//
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)
	genomeId := 3
	genomeChild, err := gnome1.mateSinglePoint(gnome2, genomeId)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 3, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 4, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")

	// check not equal sized gene pools
	//
	gene := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2],
		gnome1.Nodes[3], false), 4, 0, false)
	gnome1.Genes = append(gnome1.Genes, gene)
	genomeChild, err = gnome1.mateSinglePoint(gnome2, genomeId)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 3, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 4, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")

	// set second Genome genes to first + one more
	gnome2.Genes = append(gnome1.Genes, newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2],
		gnome1.Nodes[3], false), 4, 0, false))
	// append additional gene
	gnome2.Genes = append(gnome2.Genes, newGene(network.NewLinkWithTrait(gnome2.Traits[2], 5.5, gnome2.Nodes[1],
		gnome2.Nodes[3], true), 4, 0, false))
	genomeChild, err = gnome1.mateSinglePoint(gnome2, genomeId)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 4, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 4, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")
}

func TestGenome_mateSinglePointModular(t *testing.T) {
	rand.Seed(42)
	// Check equal sized gene pools
	//
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestModularGenome(2)
	genomeId := 3

	genomeChild, err := gnome1.mateSinglePoint(gnome2, genomeId)
	require.NoError(t, err, "failed to mate")
	require.NotNil(t, genomeChild, "Failed to create child genome")

	assert.Len(t, genomeChild.Genes, 6, "wrong number of genes")
	assert.Len(t, genomeChild.Nodes, 7, "wrong number of nodes")
	assert.Len(t, genomeChild.Traits, 3, "wrong number of traits")
	assert.Len(t, genomeChild.ControlGenes, 1, "wrong number of control genes")
}
