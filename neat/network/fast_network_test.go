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
	data = append(data, 1.0) // BIAS is a third object
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
	data = append(data, 1.0) // BIAS is third object
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

func TestFastModularNetworkSolver_Relax(t *testing.T) {
	netw := buildModularNetwork()

	// create network solver
	data := []float64{1.5, 2.0} // bias inherent
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
	data = append(data, 1.0) // BIAS is third object
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

	// do relaxation
	if res, err := fmm.Relax(steps, 1); err != nil {
		t.Error(err)
	} else if !res {
		t.Error("failed to relax within given maximal steps number")
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

func TestFastModularNetworkSolver_Flush(t *testing.T) {
	netw := buildModularNetwork()

	// create network solver
	data := []float64{1.5, 2.0} // bias inherent
	fmm, err := netw.FastNetworkSolver()
	if err != nil {
		t.Error(err)
		return
	}
	if err := fmm.LoadSensors(data); err != nil {
		t.Error(err)
		return
	}

	fmm_impl := fmm.(*FastModularNetworkSolver)
	// test has active signals
	active := countActiveSignals(fmm_impl)
	if active == 0 {
		t.Error("no active signal found")
	}

	// flush and test
	if res, err := fmm.Flush(); err != nil {
		t.Error(err)
	} else if !res {
		t.Error("failed to flush network")
	} else {
		active = countActiveSignals(fmm_impl)
		if active != 0 {
			t.Error("after flush the active signal still present", active)
		}
	}
}

func TestFastModularNetworkSolver_NodeCount(t *testing.T) {
	netw := buildModularNetwork()

	// create network solver
	if fmm, err := netw.FastNetworkSolver(); err != nil {
		t.Error(err)
	} else if fmm.NodeCount() != 9 {
		t.Error("fmm.NodeCount() != 9", fmm.NodeCount())
	}
}

func TestFastModularNetworkSolver_LinkCount(t *testing.T) {
	netw := buildModularNetwork()

	// create network solver
	if fmm, err := netw.FastNetworkSolver(); err != nil {
		t.Error(err)
	} else if fmm.LinkCount() != 9 {
		t.Error("fmm.LinkCount() != 9", fmm.LinkCount())
	}
}

func countActiveSignals(fmm_impl *FastModularNetworkSolver) int {
	active := 0
	for i := fmm_impl.biasNeuronCount; i < fmm_impl.totalNeuronCount; i++ {
		if fmm_impl.neuronSignals[i] != 0.0 {
			active++
		}
	}
	return active
}
