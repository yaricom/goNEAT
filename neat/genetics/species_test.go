package genetics

import (
	"math/rand"
	"testing"
	"github.com/yaricom/goNEAT/neat"
	"sort"
	"bytes"
)

func buildSpeciesWithOrganisms(id int) (*Species, error) {
	gen := buildTestGenome(1)

	sp := NewSpecies(id)
	for i := 0; i < 3; i++ {
		org, err := NewOrganism( float64(i + 1) * 5.0 * float64(id), gen, id)
		if err != nil {
			return nil, err
		}
		sp.addOrganism(org)
	}

	return sp, nil
}

func TestSpecies_Write(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)

	if err != nil {
		t.Error(err)
		return
	}

	out_buf := bytes.NewBufferString("")
	sp.Write(out_buf)
}

// Tests Species adjustFitness
func TestSpecies_adjustFitness(t *testing.T)  {
	sp, err := buildSpeciesWithOrganisms(1)

	if err != nil {
		t.Error(err)
		return
	}

	// Configuration
	conf := neat.NeatContext{
		DropOffAge:5,
		SurvivalThresh:0.5,
		AgeSignificance:0.5,
	}
	sp.adjustFitness(&conf)

	// test results
	if sp.Organisms[0].isChampion != true {
		t.Error("sp.Organisms[0].IsChampion", true, sp.Organisms[0].isChampion)
	}
	if sp.AgeOfLastImprovement != 1 {
		t.Error("sp.AgeOfLastImprovement", 1, sp.AgeOfLastImprovement)
	}
	if sp.MaxFitnessEver != 15.0 {
		t.Error("sp.MaxFitnessEver", 15.0, sp.MaxFitnessEver)
	}
	if sp.Organisms[2].toEliminate != true {
		t.Error("sp.Organisms[2].ToEliminate", true, sp.Organisms[2].toEliminate)
	}
}

// Tests Species countOffspring
func TestSpecies_countOffspring(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)
	if err != nil {
		t.Error(err)
		return
	}

	for i, o := range sp.Organisms {
		o.ExpectedOffspring = float64(i) * 1.5
	}

	expectedOffspring, skim := sp.countOffspring(0.5)
	sp.ExpectedOffspring = expectedOffspring

	if sp.ExpectedOffspring != 5 {
		t.Error("sp.ExpectedOffspring", 5, sp.ExpectedOffspring)
		return
	}
	if skim != 0 {
		t.Error("skim", 0, skim)
		return
	}

	sp, err = buildSpeciesWithOrganisms(2)
	if err != nil {
		t.Error(err)
		return
	}

	for i, o := range sp.Organisms {
		o.ExpectedOffspring = float64(i) * 1.5
	}
	expectedOffspring, skim = sp.countOffspring(0.4)
	sp.ExpectedOffspring = expectedOffspring
	if sp.ExpectedOffspring != 4 {
		t.Error("sp.ExpectedOffspring", 5, sp.ExpectedOffspring)
		return
	}
	if skim != 0.9 {
		t.Error("skim", 0.9, skim)
		return
	}
}

func TestSpecies_computeMaxFitness(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)
	if err != nil {
		t.Error(err)
		return
	}
	avg_check := 0.0
	for _, o := range sp.Organisms{
		avg_check += o.Fitness
	}
	avg_check /= float64(len(sp.Organisms))

	max, avg := sp.ComputeMaxAndAvgFitness()
	if max != 15.0 {
		t.Error("sp.MaxFitness != 15.0", 15.0, max)
	}
	if avg != avg_check {
		t.Error("sp.AvgFitness != avg_check", avg, avg_check)
	}
}

func TestSpecies_findChampion(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)
	if err != nil {
		t.Error(err)
		return
	}

	champ := sp.findChampion()
	if champ.Fitness != 15.0 {
		t.Error("champ.Fitness != 15.0", champ.Fitness)
	}

}

func TestSpecies_removeOrganism(t *testing.T) {
	sp, err := buildSpeciesWithOrganisms(1)
	if err != nil {
		t.Error(err)
		return
	}

	// test remove
	size := len(sp.Organisms)
	res, err := sp.removeOrganism(sp.Organisms[0])
	if res != true {
		t.Error("res != true", res, err)
	}
	if err != nil {
		t.Error("err != nil", err)
	}
	if size - 1 != len(sp.Organisms) {
		t.Error("size - 1 != len(sp.Organisms)", size - 1, len(sp.Organisms))
	}

	// test fail to remove
	size = len(sp.Organisms)
	gen := buildTestGenome(2)
	org, err := NewOrganism(6.0, gen, 1)
	if err != nil {
		t.Error(err)
		return
	}
	res, err = sp.removeOrganism(org)
	if res == true {
		t.Error("res == true", res, err)
	}
	if err == nil {
		t.Error("err == nil", res, err)
	}
	if size != len(sp.Organisms) {
		t.Error("size != len(sp.Organisms)", size, len(sp.Organisms))
	}
}

// Tests Species reproduce failure
func TestSpecies_reproduce_fail(t *testing.T) {
	sp := NewSpecies(1)

	sp.ExpectedOffspring = 1

	babies, err := sp.reproduce(1, nil, nil, nil)
	if babies != nil {
		t.Error("babies != nil")
	}
	if err == nil {
		t.Error("err == nil")
	}
}

// Tests Species reproduce success
func TestSpecies_reproduce(t *testing.T) {
	rand.Seed(42)
	in, out, nmax, n := 3, 2, 15, 3
	recurrent := false
	link_prob := 0.8

	// Configuration
	conf := neat.NeatContext {
		DropOffAge:5,
		SurvivalThresh:0.5,
		AgeSignificance:0.5,
		PopSize:30,
		CompatThreshold:0.6,
	}
	neat.LogLevel = neat.LogLevelInfo

	gen := newGenomeRand(1, in, out, n, nmax, recurrent, link_prob)
	pop, err := NewPopulation(gen, &conf)
	if err != nil {
		t.Error(err)
	}
	if pop == nil {
		t.Error("pop == nil")
	}

	// Stick the Species pointers into a new Species list for sorting
	sorted_species := make([]*Species, len(pop.Species))
	copy(sorted_species, pop.Species)

	// Sort the Species by max original fitness of its first organism
	sort.Sort(byOrganismOrigFitness(sorted_species))

	pop.Species[0].ExpectedOffspring = 11

	babies, err := pop.Species[0].reproduce(1, pop, sorted_species, &conf)
	if babies == nil {
		t.Error("No reproduction", err)
	}
	if err != nil {
		t.Error("err != nil", err)
	}

	if len(babies) != pop.Species[0].ExpectedOffspring {
		t.Error("Wrong number of babies was created", len(babies))
	}
}
