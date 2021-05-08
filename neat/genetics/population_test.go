package genetics

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat"
	"math/rand"
	"strings"
	"testing"
)

func TestNewPopulationRandom(t *testing.T) {
	rand.Seed(42)
	in, out, nmax := 3, 2, 5
	linkProb := 0.5
	conf := neat.Options{
		CompatThreshold: 0.5,
		PopSize:         10,
	}
	pop, err := NewPopulationRandom(in, out, nmax, false, linkProb, &conf)
	require.NoError(t, err, "failed to create population")
	require.NotNil(t, pop, "population expected")
	require.Len(t, pop.Organisms, conf.PopSize, "wrong population size")
	assert.EqualValues(t, 11, pop.nextNodeId, "wrong next node ID")
	assert.EqualValues(t, 101, pop.nextInnovNum, "wrong next innovation number")
	assert.True(t, len(pop.Species) > 0, "population has no species")

	for i, org := range pop.Organisms {
		assert.True(t, len(org.Genotype.Genes) > 0, "organism has no genes at: %d", i)
		assert.True(t, len(org.Genotype.Nodes) > 0, "organism has no nodes at: %d", i)
		assert.True(t, len(org.Genotype.Traits) > 0, "organism has no traits at: %d", i)
		assert.NotNil(t, org.Genotype.Phenotype, "organism has no phenotype")
	}
}

func TestNewPopulation(t *testing.T) {
	rand.Seed(42)
	in, out, nmax, n := 3, 2, 5, 3
	linkProb := 0.5
	conf := neat.Options{
		CompatThreshold: 0.5,
		PopSize:         10,
	}
	gen := newGenomeRand(1, in, out, n, nmax, false, linkProb)

	pop, err := NewPopulation(gen, &conf)
	require.NoError(t, err, "failed to create population")
	require.NotNil(t, pop, "population expected")
	require.Len(t, pop.Organisms, conf.PopSize, "wrong population size")
	lastNodeId, err := gen.getLastNodeId()
	require.NoError(t, err, "failed to get last node ID")
	assert.Equal(t, int32(lastNodeId+1), pop.nextNodeId, "wrong next node ID")

	nextGeneInnovNum, err := gen.getNextGeneInnovNum()
	require.NoError(t, err, "failed to get next gene innovation number")
	assert.Equal(t, nextGeneInnovNum-1, pop.nextInnovNum, "wrong next innovation number in population")
	require.Len(t, pop.Species, 1, "wrong species number")

	for i, org := range pop.Organisms {
		assert.True(t, len(org.Genotype.Genes) > 0, "organism has no genes at: %d", i)
		assert.True(t, len(org.Genotype.Nodes) > 0, "organism has no nodes at: %d", i)
		assert.True(t, len(org.Genotype.Traits) > 0, "organism has no traits at: %d", i)
		assert.NotNil(t, org.Genotype.Phenotype, "organism has no phenotype")
	}
}

func TestPopulation_verify(t *testing.T) {
	// first create population
	popStr := "genomestart 1\n" +
		"trait 1 0.1 0 0 0 0 0 0 0\n" +
		"trait 2 0.2 0 0 0 0 0 0 0\n" +
		"trait 3 0.3 0 0 0 0 0 0 0\n" +
		"node 1 0 1 1\n" +
		"node 2 0 1 1\n" +
		"node 3 0 1 3\n" +
		"node 4 0 0 2\n" +
		"gene 1 1 4 1.5 false 1 0 true\n" +
		"gene 2 2 4 2.5 false 2 0 true\n" +
		"gene 3 3 4 3.5 false 3 0 true\n" +
		"genomeend 1\n" +
		"genomestart 2\n" +
		"trait 1 0.1 0 0 0 0 0 0 0\n" +
		"trait 2 0.2 0 0 0 0 0 0 0\n" +
		"trait 3 0.3 0 0 0 0 0 0 0\n" +
		"node 1 0 1 1\n" +
		"node 2 0 1 1\n" +
		"node 3 0 1 3\n" +
		"node 4 0 0 2\n" +
		"gene 1 1 4 1.5 false 1 0 true\n" +
		"gene 2 2 4 2.5 false 2 0 true\n" +
		"gene 3 3 4 3.5 false 3 0 true\n" +
		"genomeend 2\n"
	conf := neat.Options{
		CompatThreshold: 0.5,
	}
	pop, err := ReadPopulation(strings.NewReader(popStr), &conf)
	require.NoError(t, err, "failed to create population")
	require.NotNil(t, pop, "population expected")

	// then verify created
	res, err := pop.Verify()
	require.NoError(t, err, "failed to verify population")
	assert.True(t, res, "Population verification failed, but must not")
}
