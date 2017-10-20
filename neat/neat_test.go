package neat

import (
	"testing"
	"os"
)

func TestLoadContext(t *testing.T) {
	config, err := os.Open("../data/p2nv.neat")
	if err != nil {
		t.Error("Failed to open config file", err)
	}

	// Load Neat Context
	nc := LoadContext(config)
	//t.Log(nc)
	if nc.TraitParamMutProb != 0.5 {
		t.Error("nc.TraitParamMutProb != 0.5")
	}
	if nc.TraitMutationPower != 1.0 {
		t.Error("nc.TraitMutationPower != 1.0")
	}
	if nc.LinkTraitMutSig != 1.0 {
		t.Error("nc.LinkTraitMutSig != 1.0")
	}
	if nc.NodeTraitMutSig != 0.5 {
		t.Error("nc.NodeTraitMutSig != 0.5 ")
	}
	if nc.WeightMutPower != 1.8 {
		t.Error("nc.WeightMutPower != 1.8")
	}
	if nc.recurProb != 0.05 {
		t.Error("nc.recurProb != 0.05")
	}
	if nc.DisjointCoeff != 1.0 {
		t.Error("nc.DisjointCoeff != 1.0")
	}
	if nc.ExcessCoeff != 1.0 {
		t.Error("nc.ExcessCoeff != 1.0")
	}
	if nc.MutdiffCoeff != 3.0 {
		t.Error("nc.MutdiffCoeff != 3.0")
	}
	if nc.CompatThreshold != 4.0 {
		t.Error("CompatThreshold")
	}
	if nc.AgeSignificance != 1.0 {
		t.Error("AgeSignificance")
	}
	if nc.SurvivalThresh != 0.4 {
		t.Error("SurvivalThresh")
	}
	if nc.MutateOnlyProb != 0.25 {
		t.Error("MutateOnlyProb")
	}
	if nc.MutateRandomTraitProb != 0.1 {
		t.Error("MutateRandomTraitProb")
	}
	if nc.MutateLinkTraitProb != 0.1 {
		t.Error("MutateLinkTraitProb")
	}
	if nc.MutateNodeTraitProb != 0.1 {
		t.Error("MutateNodeTraitProb")
	}
	if nc.MutateLinkWeightsProb != 0.8 {
		t.Error("MutateLinkWeightsProb")
	}
	if nc.MutateToggleEnableProb != 0.1 {
		t.Error("MutateToggleEnableProb")
	}
	if nc.MutateGeneReenableProb != 0.05 {
		t.Error("MutateGeneReenableProb")
	}
	if nc.MutateAddNodeProb != 0.01 {
		t.Error("MutateAddNodeProb")
	}
	if nc.MutateAddLinkProb != 0.3 {
		t.Error("MutateAddLinkProb")
	}
	if nc.MutateConnectSensors != 0.5 {
		t.Error("MutateConnectInputs")
	}
	if nc.InterspeciesMateRate != 0.001 {
		t.Error("InterspeciesMateRate")
	}
	if nc.MateMultipointProb != 0.6 {
		t.Error("MateMultipointProb")
	}
	if nc.MateMultipointAvgProb != 0.4 {
		t.Error("MateMultipointAvgProb")
	}
	if nc.MateSinglepointProb != 0.0 {
		t.Error("MateSinglepointProb")
	}
	if nc.MateOnlyProb != 0.2 {
		t.Error("MateOnlyProb")
	}
	if nc.RecurOnlyProb != 0.2 {
		t.Error("RecurOnlyProb")
	}
	if nc.PopSize != 1000 {
		t.Error("PopSize")
	}
	if nc.DropOffAge != 15 {
		t.Error("DropOffAge")
	}
	if nc.NewLinkTries != 20 {
		t.Error("NewLinkTries")
	}
	if nc.PrintEvery != 60 {
		t.Error("PrintEvery")
	}
	if nc.BabiesStolen != 0 {
		t.Error("BabiesStolen")
	}
	if nc.NumRuns != 1 {
		t.Error("NumRuns")
	}
}