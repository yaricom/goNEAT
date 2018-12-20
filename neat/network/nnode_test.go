package network

import (
	"testing"
)

// Tests NNode SensorLoad
func TestNNode_SensorLoad(t *testing.T) {
	node := NewNNode(1, InputNeuron)

	load := 21.0
	res := node.SensorLoad(load)

	if !res {
		t.Error("Failed to SensorLoad")
	}
	if node.ActivationsCount != 1 {
		t.Error("ActivationsCount", 1, node.ActivationsCount)
	}
	if node.Activation != load {
		t.Error("Activation", load, node.Activation)
	}
	if node.GetActiveOut() != load {
		t.Error("GetActiveOut", load, node.GetActiveOut())
	}

	load_2 := 36.0
	res = node.SensorLoad(load_2)
	if !res {
		t.Error("Failed to SensorLoad")
	}
	if node.ActivationsCount != 2 {
		t.Error("ActivationsCount", 2, node.ActivationsCount)
	}
	if node.Activation != load_2 {
		t.Error("Activation", load_2, node.Activation)
	}
	// Check activation and time delayed activation
	if node.GetActiveOut() != load_2 {
		t.Error("GetActiveOut", load_2, node.GetActiveOut())
	}
	if node.GetActiveOutTd() != load {
		t.Error("GetActiveOutTd", load, node.GetActiveOutTd())
	}

	// Check loading of incorrect type node
	node_n := NewNNode(1, HiddenNeuron)
	res = node_n.SensorLoad(load)
	if res {
		t.Error("Non SENSOR node can not be loaded")
	}
}

// Tests NNode AddIncoming
func TestNNode_AddIncoming(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)

	weight := 1.5
	node2.addIncoming(node, weight)
	if len(node2.Incoming) != 1 {
		t.Error("Wrong number of incoming nodes", len(node2.Incoming))
	}
	link := node2.Incoming[0]
	if link.Weight != weight {
		t.Error("Wrong incoming link weight", weight, link.Weight)
	}
	if link.InNode != node {
		t.Error("Wrong InNode in Link")
	}
	if link.OutNode != node2 {
		t.Error("Wrong OutNode in Link")
	}
}

// Tests NNode Depth
func TestNNode_Depth(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)
	node3 := NewNNode(3, OutputNeuron)

	node2.addIncoming(node, 15.0)
	node3.addIncoming(node2, 20.0)

	depth, err := node3.Depth(0)
	if err != nil {
		t.Error(err)
	}
	if depth != 2 {
		t.Error("node3.Depth", 2, depth)
	}
}

func TestNNode_DepthWithLoop(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)
	node3 := NewNNode(3, OutputNeuron)

	node2.addIncoming(node, 15.0)
	node3.addIncoming(node2, 20.0)
	node2.addIncoming(node3, 10.0)
	depth, err := node3.Depth(0)
	if err != nil {
		t.Error(err)
		return
	}
	if depth != 2 {
		t.Error("node3.Depth", 2, depth)
	}
}


// Tests NNode Flushback
func TestNNode_Flushback(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	load := 34.0
	load_2 := 14.0
	node.SensorLoad(load)
	node.SensorLoad(load_2)

	// check that node state has been updated
	if node.ActivationsCount != 2 {
		t.Error("ActivationsCount", 2, node.ActivationsCount)
	}
	if node.Activation != 14.0 {
		t.Error("Activation", load_2, node.Activation)
	}
	// Check activation and time delayed activation
	if node.GetActiveOut() != load_2 {
		t.Error("GetActiveOut", load_2, node.GetActiveOut())
	}
	if node.GetActiveOutTd() != load {
		t.Error("GetActiveOutTd", load, node.GetActiveOutTd())
	}

	// check flushback
	//
	node.Flushback()

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
