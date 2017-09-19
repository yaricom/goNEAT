package network

import (
	"testing"
	"fmt"
	"strings"
	"bytes"
	"github.com/yaricom/goNEAT/neat"
)

// Tests how NNode read working
func TestReadNNode(t *testing.T) {
	node_id, trait_id, ntype, gen_node_label := 1, 10, SENSOR, INPUT
	node_str := fmt.Sprintf("%d %d %d %d", node_id, trait_id, ntype, gen_node_label)

	trait := neat.NewTrait()
	trait.Id = 10
	traits := []*neat.Trait{trait}

	node := ReadNNode(strings.NewReader(node_str), traits)

	if node.Id != node_id {
		t.Errorf("Found node ID is not what expected, %d != %d", node_id, node.Id)
	}
	if node.Trait != trait {
		t.Error("The wrong Trait found in the node")
	}
	if node.NType != ntype {
		t.Errorf("Wrong node type found, %d != %d", ntype, node.NType)
	}
	if node.GenNodeLabel != gen_node_label {
		t.Errorf("The wrong node placement label found, %d != %d", gen_node_label, node.GenNodeLabel)
	}
}

// Tests NNode serialization
func TestWriteNNode(t *testing.T) {
	node_id, trait_id, ntype, gen_node_label := 1, 10, SENSOR, INPUT
	node_str := fmt.Sprintf("%d %d %d %d", node_id, trait_id, ntype, gen_node_label)
	trait := neat.NewTrait()
	trait.Id = 10

	node := NewNNodeInPlace(ntype, node_id, gen_node_label)
	node.Trait = trait
	out_buffer := bytes.NewBufferString("")
	node.WriteNode(out_buffer)
	out_str := out_buffer.String()

	if out_str != node_str {
		t.Errorf("Node serialization failed. Expected: %s, but found %s", node_str, out_str)
	}
}

// Tests NNode SensorLoad
func TestNNode_SensorLoad(t *testing.T) {
	node := NewNNodeInPlace(SENSOR, 1, INPUT)

	load := 21.0
	res := node.SensorLoad(load)

	if !res {
		t.Error("Failed to SensorLoad")
	}
	if node.ActivationsCount != 1 {
		t.Error("ActivationsCount", 1, node.ActivationsCount )
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
		t.Error("ActivationsCount", 2, node.ActivationsCount )
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
	node_n := NewNNodeInPlace(NEURON, 1, INPUT)
	res = node_n.SensorLoad(load)
	if res {
		t.Error("Non SENSOR node can not be loaded")
	}
}

// Tests NNode AddIncoming
func TestNNode_AddIncoming(t *testing.T) {
	node := NewNNodeInPlace(SENSOR, 1, INPUT)
	node2 := NewNNodeInPlace(NEURON, 2, HIDDEN)

	weight := 1.5
	node2.AddIncoming(node, weight)
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

// Tests NNode AddIncomingRecurrent
func TestNNode_AddIncomingRecurrent(t *testing.T) {
	node := NewNNodeInPlace(SENSOR, 1, INPUT)
	node2 := NewNNodeInPlace(NEURON, 2, HIDDEN)

	weight := 1.5
	node2.AddIncomingRecurrent(node, weight, true)
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
	if !link.IsRecurrent {
		t.Error("link.IsRecurrent", true, link.IsRecurrent)
	}
}

// Tests NNode Depth
func TestNNode_Depth(t *testing.T) {
	node := NewNNodeInPlace(SENSOR, 1, INPUT)
	node2 := NewNNodeInPlace(NEURON, 2, HIDDEN)
	node3 := NewNNodeInPlace(NEURON, 3, OUTPUT)

	node2.AddIncoming(node, 15.0)
	node3.AddIncoming(node2, 20.0)

	depth, err := node3.Depth(0)
	if err != nil {
		t.Error(err)
	}
	if depth != 2 {
		t.Error("node3.Depth", 2, depth)
	}
}

// Tests NNode Flushback
func TestNNode_Flushback(t *testing.T) {
	node := NewNNodeInPlace(SENSOR, 1, INPUT)
	load := 34.0
	load_2 := 14.0
	node.SensorLoad(load)
	node.SensorLoad(load_2)

	// check that node state has been updated
	if node.ActivationsCount != 2 {
		t.Error("ActivationsCount", 2, node.ActivationsCount )
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
