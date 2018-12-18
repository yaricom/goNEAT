package network

import "testing"

func TestFastModularNetworkSolver_RecursiveSteps(t *testing.T) {
	netw := buildNetwork()

	// Create network solver
	data := []float64{0.0, 1.0} // BIAS is 1.0 by definition
	fmm, err := netw.FastNetworkSolver()
	if err != nil {
		t.Error(err)
		return
	}
	if err := fmm.LoadSensors(data); err != nil {
		t.Error(err)
		return
	}

	// Activate objective network
	data = []float64{0.0, 1.0, 1.0} // BIAS is a third object
	if err := netw.LoadSensors(data); err != nil {
		t.Error(err)
		return
	}
	depth, err := netw.MaxDepth()
	if err != nil {
		t.Error(err)
		return
	}
	for i := 0; i < depth; i++ {
		if res, err := netw.Activate(); err != nil {
			t.Error(err)
			return
		} else if !res {
			t.Error("failed to activate")
			return
		}
	}

	// do recursive activation
	if res, err := fmm.RecursiveSteps(); err != nil {
		t.Error(err)
	} else if !res {
		t.Error("recursive acctivation retruned false")
	} else {
		outputs := fmm.ReadOutputs()
		// check results
		for i, out := range outputs {
			if out != netw.Outputs[i].Activation {
				t.Error("out != netw.Outputs[i].Activation at: ", i)
			}
		}
	}
}

func TestFastModularNetworkSolver_ForwardSteps(t *testing.T) {
	netw := buildModularNetwork()

	// create network solver
	data := []float64{1.0, 2.0} // bias inherent
	fmm, err := netw.FastNetworkSolver()
	if err != nil {
		t.Error(err)
		return
	}
	if err := fmm.LoadSensors(data); err != nil {
		t.Error(err)
		return
	}

	steps := 5

	// activate objective network
	data = []float64{1.0, 2.0, 1.0} // BIAS is third object
	if err := netw.LoadSensors(data); err != nil {
		t.Error(err)
		return
	}
	for i := 0; i < steps; i++ {
		if res, err := netw.Activate(); err != nil {
			t.Error(err)
			return
		} else if !res {
			t.Error("failed to activate")
			return
		}
	}

	// do forward steps through the solver and test results
	if res, err := fmm.ForwardSteps(steps); err != nil {
		t.Error(err)
	} else if !res {
		t.Error("forward steps retruned false")
	} else {
		outputs := fmm.ReadOutputs()
		// check results
		for i, out := range outputs {
			if out != netw.Outputs[i].Activation {
				t.Error("out != netw.Outputs[i].Activation at: ", i, out, "!=", netw.Outputs[i].Activation)
			}
		}
	}
}
