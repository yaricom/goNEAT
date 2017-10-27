package network

import (
	"testing"
)

func buildNetwork() *Network {
	all_nodes := []*NNode {
		NewNNodeInPlace(SensorNode, 1, InputNeuron),
		NewNNodeInPlace(SensorNode, 2, InputNeuron),
		NewNNodeInPlace(SensorNode, 3, InputNeuron),
		NewNNodeInPlace(NeuronNode, 4, HiddenNeuron),
		NewNNodeInPlace(NeuronNode, 5, HiddenNeuron),
		NewNNodeInPlace(NeuronNode, 6, HiddenNeuron),
		NewNNodeInPlace(NeuronNode, 7, OutputNeuron),
		NewNNodeInPlace(NeuronNode, 8, OutputNeuron),
	}

	// HIDDEN 4
	all_nodes[3].AddIncoming(all_nodes[0], 15.0)
	all_nodes[3].AddIncoming(all_nodes[1], 10.0)
	// HIDDEN 5
	all_nodes[4].AddIncoming(all_nodes[1], 5.0)
	all_nodes[4].AddIncoming(all_nodes[2], 1.0)
	// HIDDEN 6
	all_nodes[5].AddIncoming(all_nodes[4], 17.0)
	// OUTPUT 7
	all_nodes[6].AddIncoming(all_nodes[3], 7.0)
	all_nodes[6].AddIncoming(all_nodes[5], 4.5)
	// OUTPUT 8
	all_nodes[7].AddIncoming(all_nodes[5], 13.0)

	return NewNetwork(all_nodes[0:3], all_nodes[6:8], all_nodes, 0)
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
func TestNetwork_OutputIsOff(t *testing.T)  {
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
				t.Error("ActivationsCount not set", node.ActivationsCount, node )
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
	netw.Flush()
	for _, node := range netw.AllNodes() {
		if node.ActivationsCount != 0 {
			t.Error("ActivationsCount", 0, node.ActivationsCount )
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

// Test Network FlushCheck
func TestNetwork_FlushCheck(t *testing.T) {
	netw := buildNetwork()
	// activate and check state
	res, err := netw.Activate()
	if err != nil {
		t.Error(err)
	}
	if !res {
		t.Error("Failed to activate")
	}

	flush_err := netw.FlushCheck()
	if flush_err == nil {
		t.Error("Flush check expected to fail")
	}

	netw.Flush()
	flush_err = netw.FlushCheck()
	if flush_err != nil {
		t.Error("Flush check expected to succeed")
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
	nodes[4].AddIncoming(nodes[7], 3.0)

	recur = netw.IsRecurrent(nodes[5], nodes[7], &count, 32)
	if !recur {
		t.Error("Network is actually recurrent now")
	}
}
