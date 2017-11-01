package genetics

import (
	"math/rand"
	"testing"
	"github.com/yaricom/goNEAT/neat"
	"sort"
	"bytes"
)

func buildSpeciesWithOrganisms(id int) *Species {
	gen := &Genome{
		Id:1,
	}

	sp := NewSpecies(id)
	sp.addOrganism(NewOrganism(5.0 * float64(id), gen, id))
	sp.addOrganism(NewOrganism(15.0 * float64(id), gen, id))
	sp.addOrganism(NewOrganism(10.0 * float64(id), gen, id))

	return sp
}

func TestSpecies_Write(t *testing.T) {
	sp := buildSpeciesWithOrganisms(1)

	out_buf := bytes.NewBufferString("")
	sp.Write(out_buf)

	t.Log(out_buf)
}

// Tests Species adjustFitness
func TestSpecies_adjustFitness(t *testing.T)  {
	sp := buildSpeciesWithOrganisms(1)

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
	sp := buildSpeciesWithOrganisms(1)
	for i, o := range sp.Organisms {
		o.ExpectedOffspring = float64(i) * 1.5
	}

	skim := sp.countOffspring(0.5)

	if sp.ExpectedOffspring != 5 {
		t.Error("sp.ExpectedOffspring", 5, sp.ExpectedOffspring)
	}
	if skim != 0 {
		t.Error("skim", 0, skim)
	}

	sp = buildSpeciesWithOrganisms(2)
	for i, o := range sp.Organisms {
		o.ExpectedOffspring = float64(i) * 1.5
	}
	skim = sp.countOffspring(0.4)
	if sp.ExpectedOffspring != 4 {
		t.Error("sp.ExpectedOffspring", 5, sp.ExpectedOffspring)
	}
	if skim != 0.9 {
		t.Error("skim", 0.9, skim)
	}
}

func TestSpecies_computeMaxFitness(t *testing.T) {
	sp := buildSpeciesWithOrganisms(1)
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
	sp := buildSpeciesWithOrganisms(1)

	champ := sp.findChampion()
	if champ.Fitness != 15.0 {
		t.Error("champ.Fitness != 15.0", champ.Fitness)
	}

}

func TestSpecies_removeOrganism(t *testing.T) {
	sp := buildSpeciesWithOrganisms(1)

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
	gen := &Genome{
		Id:1,
	}
	org := NewOrganism(6.0, gen, 1)
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

	res, err := sp.reproduce(1, nil, nil, nil)
	if res != false {
		t.Error("res != false")
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

	gen := NewGenomeRand(1, in, out, n, nmax, recurrent, link_prob)
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
	sort.Sort(ByOrganismOrigFitness(sorted_species))

	pop.Species[0].ExpectedOffspring = 11

	res, err := pop.Species[0].reproduce(1, pop, sorted_species, &conf)
	if !res {
		t.Error("No reproduction", err)
	}
	if err != nil {
		t.Error("err != nil", err)
	}
	after := 0
	for _, sp := range pop.Species {
		after += len(sp.Organisms)
	}

	if after != conf.PopSize + pop.Species[0].ExpectedOffspring {
		t.Error("No new baby was created", after)
	}
}
