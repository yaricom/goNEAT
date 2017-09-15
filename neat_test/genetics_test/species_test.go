package genetics_test

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat"
)

// Tests Species ReadGene
func TestGene_AdjustFitness(t *testing.T)  {
	gen := &genetics.Genome{
		GenomeId:1,
	}

	sp := genetics.NewSpecies(1)
	sp.AddOrganism(genetics.NewOrganism(5.0, gen, 10))
	sp.AddOrganism(genetics.NewOrganism(15.0, gen, 10))
	sp.AddOrganism(genetics.NewOrganism(10.0, gen, 10))

	// Configuration
	conf := neat.Neat{
		DropOffAge:5,
		SurvivalThresh:0.5,
	}

	t.Log(sp.Organisms)

	sp.AdjustFitness(&conf)

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
	if len(sp.Organisms) != 2 {
		t.Error("len(sp.Organisms)", 2, len(sp.Organisms))
	}
}
