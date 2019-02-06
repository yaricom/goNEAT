package network

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/utils"
)

func buildNetwork() *Network {
	all_nodes := []*NNode{
		NewNNode(1, InputNeuron),
		NewNNode(2, InputNeuron),
		NewNNode(3, BiasNeuron),
		NewNNode(4, HiddenNeuron),
		NewNNode(5, HiddenNeuron),
		NewNNode(6, HiddenNeuron),
		NewNNode(7, OutputNeuron),
		NewNNode(8, OutputNeuron),
	}

	// HIDDEN 4
	all_nodes[3].addIncoming(all_nodes[0], 15.0)
	all_nodes[3].addIncoming(all_nodes[1], 10.0)
	// HIDDEN 5
	all_nodes[4].addIncoming(all_nodes[1], 5.0)
	all_nodes[4].addIncoming(all_nodes[2], 1.0)
	// HIDDEN 6
	all_nodes[5].addIncoming(all_nodes[4], 17.0)
	// OUTPUT 7
	all_nodes[6].addIncoming(all_nodes[3], 7.0)
	all_nodes[6].addIncoming(all_nodes[5], 4.5)
	// OUTPUT 8
	all_nodes[7].addIncoming(all_nodes[5], 13.0)

	return NewNetwork(all_nodes[0:3], all_nodes[6:8], all_nodes, 0)
}

func buildModularNetwork() *Network {
	all_nodes := []*NNode{
		NewNNode(1, InputNeuron),
		NewNNode(2, InputNeuron),
		NewNNode(3, BiasNeuron),
		NewNNode(4, HiddenNeuron),
		NewNNode(5, HiddenNeuron),
		NewNNode(7, HiddenNeuron),
		NewNNode(8, OutputNeuron),
		NewNNode(9, OutputNeuron),
	}
	control_nodes := []*NNode{
		NewNNode(6, HiddenNeuron),
	}
	// HIDDEN 6
	control_nodes[0].ActivationType = utils.MultiplyModuleActivation
	control_nodes[0].addIncoming(all_nodes[3], 1.0)
	control_nodes[0].addIncoming(all_nodes[4], 1.0)
	control_nodes[0].addOutgoing(all_nodes[5], 1.0)

	// HIDDEN 4
	all_nodes[3].ActivationType = utils.LinearActivation
	all_nodes[3].addIncoming(all_nodes[0], 15.0)
	all_nodes[3].addIncoming(all_nodes[2], 10.0)
	// HIDDEN 5
	all_nodes[4].ActivationType = utils.LinearActivation
	all_nodes[4].addIncoming(all_nodes[1], 5.0)
	all_nodes[4].addIncoming(all_nodes[2], 1.0)

	// HIDDEN 7
	all_nodes[5].ActivationType = utils.NullActivation

	// OUTPUT 8
	all_nodes[6].addIncoming(all_nodes[5], 4.5)
	all_nodes[6].ActivationType = utils.LinearActivation
	// OUTPUT 9
	all_nodes[7].addIncoming(all_nodes[5], 13.0)
	all_nodes[7].ActivationType = utils.LinearActivation

	return NewModularNetwork(all_nodes[0:3], all_nodes[6:8], all_nodes, control_nodes, 0)
}

func TestModularNetwork_Activate(t *testing.T) {
	netw := buildModularNetwork()
	data := []float64{1.0, 2.0, 0.5}
	netw.LoadSensors(data)

	for i := 0; i < 5; i++ {
		if res, err := netw.Activate(); err != nil {
			t.Error(err)
			return
		} else if !res {
			t.Error("failed to activate")
			return
		}
	}
	if netw.Outputs[0].Activation != 945 {
		t.Error("netw.Outputs[0].Activation != 945", netw.Outputs[0].Activation)
	}
	if netw.Outputs[1].Activation != 2730 {
		t.Error("netw.Outputs[1].Activation != 2730", netw.Outputs[1].Activation)
	}
}

// Tests Network MaxDepth
func TestNetwork_MaxDepth(t *testing.T) {
	netw := buildNetwork()

	depth, err := netw.MaxDepth()
	if err != nil {
		t.Error(err)
	}
	if depth != 3 {
		t.Error("MaxDepth", 3, depth)
	}
}

// Tests Network OutputIsOff
func TestNetwork_OutputIsOff(t *testing.T) {
	netw := buildNetwork()

	res := netw.OutputIsOff()
	if !res {
		t.Error("OutputIsOff", res)
	}
}

// Tests Network Activate
func TestNetwork_Activate(t *testing.T) {
	netw := buildNetwork()

	res, err := netw.Activate()
	if err != nil {
		t.Error(err)
	}
	if !res {
		t.Error("Failed to activate")
	}
	// check activation
	for _, node := range netw.AllNodes() {
		if node.IsNeuron() {
			if node.ActivationsCount == 0 {
				t.Error("ActivationsCount not set", node.ActivationsCount, node)
			}
			if node.Activation == 0 {
				t.Error("Activation not set", node.Activation, node)
			}
			// Check activation and time delayed activation
			if node.GetActiveOut() == 0 {
				t.Error("GetActiveOut not set", node.GetActiveOut(), node)
			}
		}
	}
}

// Test Network LoadSensors
func TestNetwork_LoadSensors(t *testing.T) {
	netw := buildNetwork()

	sensors := []float64{1.0, 3.4, 5.6}

	netw.LoadSensors(sensors)
	counter := 0
	for _, node := range netw.AllNodes() {
		if node.IsSensor() {
			if node.Activation != sensors[counter] {
				t.Error("Sensor value wrong", sensors[counter], node.Activation)
			}
			if node.ActivationsCount != 1 {
				t.Error("Sensor activations count wrong", 1, node.ActivationsCount)
			}
			counter++
		}
	}
}

// Test Network Flush
func TestNetwork_Flush(t *testing.T) {
	netw := buildNetwork()

	// activate and check state
	res, err := netw.Activate()
	if err != nil {
		t.Error(err)
	}
	if !res {
		t.Error("Failed to activate")
	}
	netw.Activate()

	// flush and check
	res, err = netw.Flush()
	if err != nil {
		t.Error(err)
		return
	}
	if !res {
		t.Error("Network flush failed")
		return
	}

	for _, node := range netw.AllNodes() {
		if node.ActivationsCount != 0 {
			t.Error("ActivationsCount", 0, node.ActivationsCount)
		}
		if node.Activation != 0 {
			t.Error("Activation", 0, node.Activation)
		}
		// Check activation and time delayed activation
		if node.GetActiveOut() != 0 {
			t.Error("GetActiveOut", 0, node.GetActiveOut())
		}
		if node.GetActiveOutTd() != 0 {
			t.Error("GetActiveOutTd", 0, node.GetActiveOutTd())
		}
	}
}

// Tests Network NodeCount
func TestNetwork_NodeCount(t *testing.T) {
	netw := buildNetwork()

	count := netw.NodeCount()
	if count != 8 {
		t.Error("Wrong network's node count", 8, count)
	}
}

// Tests Network LinkCount
func TestNetwork_LinkCount(t *testing.T) {
	netw := buildNetwork()

	count := netw.LinkCount()
	if count != 8 {
		t.Error("Wrong network's link count", 8, count)
	}
}

// Tests Network IsRecurrent
func TestNetwork_IsRecurrent(t *testing.T) {
	netw := buildNetwork()

	nodes := netw.AllNodes()

	count := 0
	recur := netw.IsRecurrent(nodes[0], nodes[7], &count, 32)
	if recur {
		t.Error("Network is not recurrent")
	}

	// Introduce recurrence
	nodes[4].addIncoming(nodes[7], 3.0)

	recur = netw.IsRecurrent(nodes[5], nodes[7], &count, 32)
	if !recur {
		t.Error("Network is actually recurrent now")
	}
}

// test fast network solver generation
func TestNetwork_FastNetworkSolver(t *testing.T) {
	netw := buildModularNetwork()

	solver, err := netw.FastNetworkSolver()
	if err != nil {
		t.Error(err)
		return
	}

	// check solver
	if solver.NodeCount() != netw.NodeCount() {
		t.Error("solver.NodeCount() != netw.NodeCount()", solver.NodeCount(), netw.NodeCount())
	}
	if solver.LinkCount() != netw.LinkCount() {
		t.Error("solver.LinkCount() != netw.LinkCount()", solver.LinkCount(), netw.LinkCount())
	}
}
