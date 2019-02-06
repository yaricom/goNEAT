package neat

import (
	"testing"
	"os"
	"fmt"
	"github.com/yaricom/goNEAT/neat/utils"
)

func TestLoadContext(t *testing.T) {
	config, err := os.Open("../data/xor_test.neat")
	if err != nil {
		t.Error("Failed to open config file", err)
	}

	// Load Neat Context
	nc := LoadContext(config)
	checkNeatContext(nc, t)
}

func TestNeatContext_LoadContext(t *testing.T) {
	config, err := os.Open("../data/xor_test.neat.yml")
	if err != nil {
		t.Error("Failed to open config file", err)
	}

	// Load YAML context
	nc := NewNeatContext()
	err = nc.LoadContext(config)
	if err != nil {
		t.Error(err)
	}

	checkNeatContext(nc, t)

	// check activators
	if len(nc.NodeActivators) != 4 {
		t.Error(fmt.Sprintf("len(nc.NodeActivators) != 4, but: %d", len(nc.NodeActivators)))
		return
	}
	activators := []utils.NodeActivationType{utils.SigmoidBipolarActivation,
		utils.GaussianBipolarActivation, utils.LinearAbsActivation, utils.SineActivation}
	probs := []float64{0.25, 0.35, 0.15, 0.25}
	for i, a := range activators {
		if nc.NodeActivators[i] != a {
			t.Error(fmt.Sprintf("Wrong CPPN activator type at: %d", i))
		}
		if nc.NodeActivatorsProb[i] != probs[i] {
			t.Error("nc.NodeActivatorsProb[i] != probs[i]", nc.NodeActivatorsProb[i], probs[i])
		}
	}
}

func checkNeatContext(nc *NeatContext, t *testing.T) {
	if nc.TraitParamMutProb != 0.5 {
		t.Error("nc.TraitParamMutProb != 0.5", nc.TraitParamMutProb)
	}
	if nc.TraitMutationPower != 1.0 {
		t.Error("nc.TraitMutationPower != 1.0", nc.TraitMutationPower)
	}
	if nc.WeightMutPower != 2.5 {
		t.Error("nc.WeightMutPower != 2.5", nc.WeightMutPower != 2.5)
	}
	if nc.DisjointCoeff != 1.0 {
		t.Error("nc.DisjointCoeff != 1.0", nc.DisjointCoeff)
	}
	if nc.ExcessCoeff != 1.0 {
		t.Error("nc.ExcessCoeff != 1.0", nc.ExcessCoeff)
	}
	if nc.MutdiffCoeff != 0.4 {
		t.Error("nc.MutdiffCoeff", nc.MutdiffCoeff)
	}
	if nc.CompatThreshold != 3.0 {
		t.Error("CompatThreshold", nc.CompatThreshold)
	}
	if nc.AgeSignificance != 1.0 {
		t.Error("AgeSignificance", nc.AgeSignificance)
	}
	if nc.SurvivalThresh != 0.2 {
		t.Error("SurvivalThresh", nc.SurvivalThresh)
	}
	if nc.MutateOnlyProb != 0.25 {
		t.Error("MutateOnlyProb", nc.MutateOnlyProb)
	}
	if nc.MutateRandomTraitProb != 0.1 {
		t.Error("MutateRandomTraitProb", nc.MutateRandomTraitProb)
	}
	if nc.MutateLinkTraitProb != 0.1 {
		t.Error("MutateLinkTraitProb", nc.MutateLinkTraitProb)
	}
	if nc.MutateNodeTraitProb != 0.1 {
		t.Error("MutateNodeTraitProb", nc.MutateNodeTraitProb)
	}
	if nc.MutateLinkWeightsProb != 0.9 {
		t.Error("MutateLinkWeightsProb", nc.MutateLinkWeightsProb)
	}
	if nc.MutateToggleEnableProb != 0.0 {
		t.Error("MutateToggleEnableProb", nc.MutateToggleEnableProb)
	}
	if nc.MutateGeneReenableProb != 0.0 {
		t.Error("MutateGeneReenableProb", nc.MutateGeneReenableProb)
	}
	if nc.MutateAddNodeProb != 0.03 {
		t.Error("MutateAddNodeProb", nc.MutateAddNodeProb)
	}
	if nc.MutateAddLinkProb != 0.08 {
		t.Error("MutateAddLinkProb", nc.MutateAddLinkProb)
	}
	if nc.MutateConnectSensors != 0.5 {
		t.Error("MutateConnectSensors", nc.MutateConnectSensors)
	}
	if nc.InterspeciesMateRate != 0.001 {
		t.Error("InterspeciesMateRate", nc.InterspeciesMateRate)
	}
	if nc.MateMultipointProb != 0.3 {
		t.Error("MateMultipointProb", nc.MateMultipointProb)
	}
	if nc.MateMultipointAvgProb != 0.3 {
		t.Error("MateMultipointAvgProb", nc.MateMultipointAvgProb)
	}
	if nc.MateSinglepointProb != 0.3 {
		t.Error("MateSinglepointProb", nc.MateSinglepointProb)
	}
	if nc.MateOnlyProb != 0.2 {
		t.Error("MateOnlyProb", nc.MateOnlyProb)
	}
	if nc.RecurOnlyProb != 0.0 {
		t.Error("RecurOnlyProb", nc.RecurOnlyProb)
	}
	if nc.PopSize != 200 {
		t.Error("PopSize", nc.PopSize)
	}
	if nc.DropOffAge != 50 {
		t.Error("DropOffAge", nc.DropOffAge)
	}
	if nc.NewLinkTries != 50 {
		t.Error("NewLinkTries", nc.NewLinkTries)
	}
	if nc.PrintEvery != 10 {
		t.Error("PrintEvery", nc.PrintEvery)
	}
	if nc.BabiesStolen != 0 {
		t.Error("BabiesStolen", nc.BabiesStolen)
	}
	if nc.NumRuns != 100 {
		t.Error("NumRuns", nc.NumRuns)
	}
	if nc.NumGenerations != 100 {
		t.Error("NumGenerations", nc.NumGenerations)
	}
	if nc.EpochExecutorType != 0 {
		t.Error("EpochExecutorType", nc.EpochExecutorType)
	}
	if nc.GenCompatMethod != 1 {
		t.Error("GenCompatMethod", nc.GenCompatMethod)
	}
}