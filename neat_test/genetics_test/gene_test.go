package genetics_test

import (
	"testing"
	"fmt"
	"github.com/yaricom/goNEAT/neat/genetics"
	"strings"
	"github.com/yaricom/goNEAT/neat/network"
	"bytes"
)

// Tests Gene ReadGene
func TestGene_ReadGene(t *testing.T)  {
	// gene  1 1 4 1.1983046913458986 0 1.0 1.1983046913458986 0
	traitId, inNodeId, outNodeId := 1, 1, 4
	innov_num := int64(1)
	weight, mut_num := 1.1983046913458986, 1.1983046913458986
	recurrent, enabled := false, false
	gene_str := fmt.Sprintf("gene %d %d %d %g %t %d %g %t",
		traitId, inNodeId, outNodeId, weight, recurrent, innov_num, mut_num, enabled)

	trait := network.NewTrait()
	trait.TraitId = 1
	nodes := []*network.NNode{
		network.NewNNodeInPlace(network.SENSOR, 1, network.INPUT),
		network.NewNNodeInPlace(network.NEURON, 4, network.HIDDEN),
	}

	gene := genetics.ReadGene(strings.NewReader(gene_str), []*network.Trait{trait}, nodes)

	if gene.InnovationNum != innov_num {
		t.Error("gene.InnovationNum", innov_num, gene.InnovationNum)
	}
	if gene.MutationNum != mut_num {
		t.Error("gene.MutationNum", mut_num, gene.MutationNum)
	}
	if gene.IsEnabled != enabled {
		t.Error("gene.IsEnabled", enabled, gene.IsEnabled)
	}
	link := gene.Link
	if link.LinkTrait.TraitId != traitId {
		t.Error("link.LinkTrait.TraitId", traitId, link.LinkTrait.TraitId)
	}
	if link.InNode.NodeId != inNodeId {
		t.Error("link.InNode.NodeId", inNodeId, link.InNode.NodeId)
	}
	if link.OutNode.NodeId != outNodeId {
		t.Error("link.OutNode.NodeId", outNodeId, link.OutNode.NodeId)
	}
	if link.Weight != weight {
		t.Error("link.Weight", weight, link.Weight)
	}
	if link.IsRecurrent != recurrent {
		t.Error("link.IsRecurrent", recurrent, link.IsRecurrent)
	}
}

// Tests Gene WriteGene
func TestGene_WriteGene(t *testing.T)  {
	// gene  1 1 4 1.1983046913458986 0 1.0 1.1983046913458986 0
	traitId, inNodeId, outNodeId := 1, 1, 4
	innov_num := int64(1)
	weight, mut_num := 1.1983046913458986, 1.1983046913458986
	recurrent, enabled := false, false
	gene_str := fmt.Sprintf("gene %d %d %d %g %t %d %g %t",
		traitId, inNodeId, outNodeId, weight, recurrent, innov_num, mut_num, enabled)

	trait := network.NewTrait()
	trait.TraitId = traitId
	gene := genetics.NewGeneWithTrait(trait, weight, network.NewNNodeInPlace(network.SENSOR, 1, network.INPUT),
		network.NewNNodeInPlace(network.NEURON, 4, network.HIDDEN), recurrent, innov_num, mut_num)
	gene.IsEnabled = enabled

	out_buf := bytes.NewBufferString("")
	gene.WriteGene(out_buf)

	out_str := out_buf.String()
	if gene_str != out_str {
		t.Errorf("Wrong Gene serialization\n[%s]\n[%s]", gene_str, out_str)
	}
}
