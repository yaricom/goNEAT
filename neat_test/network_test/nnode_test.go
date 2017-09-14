package network_test

import (
	"testing"
	"fmt"
	"strings"
	"github.com/yaricom/goNEAT/neat/network"
)

// Tests how NNode read working
func TestReadNNode(t *testing.T) {
	node_id, trait_id, ntype, gen_node_label := 1, 10, network.SENSOR, network.INPUT
	node_str := fmt.Sprintf("node %d %d %d %d", node_id, trait_id, ntype, gen_node_label)

	trait := network.NewTraitWithParams(10, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
	traits := []*network.Trait{trait}

	node := network.ReadNNode(strings.NewReader(node_str), traits)

	if node.NodeId != node_id {
		t.Errorf("Found node ID is not what expected, %d != %d", node_id, node.NodeId)
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