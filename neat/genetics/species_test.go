package genetics

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat"
	"math/rand"
	"sort"
	"testing"
)

func buildSpeciesWithOrganisms(id int) (*Species, error) {
	gen := buildTestGenome(1)

	sp := NewSpecies(id)
	for i := 0; i < 3; i++ {
		org, err := NewOrganism(float64(i+1)*5.0*float64(id), gen, id)
		if err != nil {
			return nil, err
		}
		sp.addOrganism(org)
	}

	return sp, nil
}

func TestSpecies_Write(t *testing.T) {
	spStr := `/* Species #1 : (Size 3) (AF 10.000) (Age 1)  */
        /* Organism #1 Fitness: 15.000 Error: 0.000 */
        genomestart 1
        trait 1 0.1 0 0 0 0 0 0 0
        trait 3 0.3 0 0 0 0 0 0 0
        trait 2 0.2 0 0 0 0 0 0 0
        node 1 0 1 1 NullActivation
        node 2 0 1 1 NullActivation
        node 3 0 1 3 SigmoidSteepenedActivation
        node 4 0 0 2 SigmoidSteepenedActivation
        gene 1 1 4 1.5 false 1 0 true
        gene 2 2 4 2.5 false 2 0 true
        gene 3 3 4 3.5 false 3 0 true
        genomeend 1
        /* Organism #1 Fitness: 10.000 Error: 0.000 */
        genomestart 1
        trait 1 0.1 0 0 0 0 0 0 0
        trait 3 0.3 0 0 0 0 0 0 0
        trait 2 0.2 0 0 0 0 0 0 0
        node 1 0 1 1 NullActivation
        node 2 0 1 1 NullActivation
        node 3 0 1 3 SigmoidSteepenedActivation
        node 4 0 0 2 SigmoidSteepenedActivation
        gene 1 1 4 1.5 false 1 0 true
        gene 2 2 4 2.5 false 2 0 true
        gene 3 3 4 3.5 false 3 0 true
        genomeend 1
        /* Organism #1 Fitness: 5.000 Error: 0.000 */
        genomestart 1
        trait 1 0.1 0 0 0 0 0 0 0
        trait 3 0.3 0 0 0 0 0 0 0
        trait 2 0.2 0 0 0 0 0 0 0
        node 1 0 1 1 NullActivation
        node 2 0 1 1 NullActivation
        node 3 0 1 3 SigmoidSteepenedActivation
        node 4 0 0 2 SigmoidSteepenedActivation
        gene 1 1 4 1.5 false 1 0 true
        gene 2 2 4 2.5 false 2 0 true
        gene 3 3 4 3.5 false 3 0 true
        genomeend 1`

	sp, err := buildSpeciesWithOrganisms(1)
	require.NoError(t, err, "failed to build species")

	outBuf := bytes.NewBufferString("")
	err = sp.Write(outBuf)
	require.NoError(t, err)

	_, inputTokens, err := bufio.ScanLines([]byte(spStr), true)
	require.NoError(t, err, "failed to parse input string")
	_, outputTokens, err := bufio.ScanLines(outBuf.Bytes(), true)
	require.NoError(t, err, "failed to parse output string")

	for i, gsr := range inputTokens {
		assert.Equal(t, gsr, outputTokens[i], "lines mismatch at: %d", i)
	}
}

// Tests Species adjustFitness
func TestSpecies_adjustFitness(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)
	require.NoError(t, err, "failed to build species")

	// Configuration
	conf := neat.Options{
		DropOffAge:      5,
		SurvivalThresh:  0.5,
		AgeSignificance: 0.5,
	}
	sp.adjustFitness(&conf)

	// test results
	assert.True(t, sp.Organisms[0].isChampion)
	assert.Equal(t, 1, sp.AgeOfLastImprovement)
	assert.Equal(t, 15.0, sp.MaxFitnessEver)
	assert.True(t, sp.Organisms[2].toEliminate)
}

// Tests Species countOffspring
func TestSpecies_countOffspring(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)
	require.NoError(t, err, "failed to build species")

	for i, o := range sp.Organisms {
		o.ExpectedOffspring = float64(i) * 1.5
	}

	expectedOffspring, skim := sp.countOffspring(0.5)
	sp.ExpectedOffspring = expectedOffspring

	assert.Equal(t, 5, sp.ExpectedOffspring, "wrong number of expected ofsprings")
	assert.EqualValues(t, 0, skim, "wrong skim value")

	// Build another species and test
	//
	sp, err = buildSpeciesWithOrganisms(2)
	require.NoError(t, err, "failed to build species")

	for i, o := range sp.Organisms {
		o.ExpectedOffspring = float64(i) * 1.5
	}
	expectedOffspring, skim = sp.countOffspring(0.4)
	sp.ExpectedOffspring = expectedOffspring
	assert.Equal(t, 4, sp.ExpectedOffspring, "wrong number of expected ofsprings")
	assert.EqualValues(t, 0.9, skim, "wrong skim value")
}

func TestSpecies_computeMaxFitness(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)
	require.NoError(t, err, "failed to build species")
	avgCheck := 0.0
	for _, o := range sp.Organisms {
		avgCheck += o.Fitness
	}
	avgCheck /= float64(len(sp.Organisms))

	max, avg := sp.ComputeMaxAndAvgFitness()
	assert.Equal(t, 15.0, max, "wrong max fitness")
	assert.Equal(t, avgCheck, avg, "wrong avg fitness")
}

func TestSpecies_findChampion(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)
	require.NoError(t, err, "failed to build species")

	champ := sp.findChampion()
	assert.Equal(t, 15.0, champ.Fitness, "wrong champion's fitness")
}

func TestSpecies_removeOrganism(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)
	require.NoError(t, err, "failed to build species")

	// test remove
	size := len(sp.Organisms)
	res, err := sp.removeOrganism(sp.Organisms[0])
	require.NoError(t, err, "failed to remove organism")
	require.True(t, res, "organism removal failed")
	require.Len(t, sp.Organisms, size-1, "wrong number of organisms after removal")

	// test fail to remove
	//
	size = len(sp.Organisms)
	gen := buildTestGenome(2)
	org, err := NewOrganism(6.0, gen, 1)
	require.NoError(t, err, "failed to create organism")
	res, err = sp.removeOrganism(org)
	assert.False(t, res, "not existing organism can not be removed")
	assert.EqualError(t, err, fmt.Sprintf("attempt to remove nonexistent Organism from Species with #of organisms: %d", size))
	require.Len(t, sp.Organisms, size, "wrong number of organisms in species after unsuccessful removal attempt")
}

// Tests Species reproduce failure
func TestSpecies_reproduce_fail(t *testing.T) {
	sp := NewSpecies(1)

	sp.ExpectedOffspring = 1

	babies, err := sp.reproduce(1, nil, nil, nil)
	assert.Empty(t, babies, "no offsprings expected")
	assert.EqualError(t, err, "attempt to reproduce out of empty species")
}

// Tests Species reproduce success
func TestSpecies_reproduce(t *testing.T) {
	rand.Seed(42)
	in, out, nmax, n := 3, 2, 15, 3
	linkProb := 0.8

	// Configuration
	conf := neat.Options{
		DropOffAge:      5,
		SurvivalThresh:  0.5,
		AgeSignificance: 0.5,
		PopSize:         30,
		CompatThreshold: 0.6,
	}
	neat.LogLevel = neat.LogLevelInfo

	gen := newGenomeRand(1, in, out, n, nmax, false, linkProb)
	pop, err := NewPopulation(gen, &conf)
	require.NoError(t, err, "failed to create population")
	require.NotNil(t, pop, "population expected")

	// Stick the Species pointers into a new Species list for sorting
	sortedSpecies := make([]*Species, len(pop.Species))
	copy(sortedSpecies, pop.Species)

	// Sort the Species by max original fitness of its first organism
	sort.Sort(byOrganismOrigFitness(sortedSpecies))

	pop.Species[0].ExpectedOffspring = 11

	babies, err := pop.Species[0].reproduce(1, pop, sortedSpecies, &conf)
	require.NoError(t, err, "failed to reproduce")
	require.NotEmpty(t, babies, "offsprings expected")

	assert.Len(t, babies, pop.Species[0].ExpectedOffspring, "Wrong number of babies was created")
}
