package genetics

import (
	"testing"
	"github.com/yaricom/goNEAT/neat"
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

// Tests Species ReadGene
func TestGene_adjustFitness(t *testing.T)  {
	sp := buildSpeciesWithOrganisms(1)

	// Configuration
	conf := neat.Neat{
		DropOffAge:5,
		SurvivalThresh:0.5,
		AgeSignificance:0.5,
	}
	sp.adjustFitness(&conf)

	// test results
	if sp.Organisms[0].IsChampion != true {
		t.Error("sp.Organisms[0].IsChampion", true, sp.Organisms[0].IsChampion)
	}
	if sp.AgeOfLastImprovement != 1 {
		t.Error("sp.AgeOfLastImprovement", 1, sp.AgeOfLastImprovement)
	}
	if sp.MaxFitnessEver != 15.0 {
		t.Error("sp.MaxFitnessEver", 15.0, sp.MaxFitnessEver)
	}
	if sp.Organisms[2].ToEliminate != true {
		t.Error("sp.Organisms[2].ToEliminate", true, sp.Organisms[2].ToEliminate)
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

// Tests Species computeAvgFitness
func TestSpecies_computeAvgFitness(t *testing.T) {
	sp := buildSpeciesWithOrganisms(1)

	avg_check := 0.0
	for _, o := range sp.Organisms{
		avg_check += o.Fitness
	}
	avg_check /= float64(len(sp.Organisms))

	sp.computeAvgFitness()
	if sp.AvgFitness != avg_check {
		t.Error("sp.AvgFitness != avg_check", sp.AvgFitness, avg_check)
	}
}

func TestSpecies_computeMaxFitness(t *testing.T) {
	sp := buildSpeciesWithOrganisms(1)

	sp.computeMaxFitness()
	if sp.MaxFitness != 15.0 {
		t.Error("sp.MaxFitness != 15.0", 15.0, sp.MaxFitness)
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
// TODO do this
/*
func TestSpecies_reproduce(t *testing.T) {
	sp := buildSpeciesWithOrganisms(1)
	sp.ExpectedOffspring = 2

	sorted_species := []*Species {
		buildSpeciesWithOrganisms(4),
		buildSpeciesWithOrganisms(3),
		buildSpeciesWithOrganisms(2),
	}

	// Configuration
	conf := neat.Neat{
		DropOffAge:5,
		SurvivalThresh:0.5,
		AgeSignificance:0.5,
		PopSize:3,
	}

	res, err := sp.reproduce(1, nil, sorted_species, &conf)
	if !res {
		t.Error("!res", err)
	}
	if err != nil {
		t.Error("err != nil", err)
	}
	if len(sp.Organisms) != 4 {
		t.Error("No new baby was created")
	}
}*/