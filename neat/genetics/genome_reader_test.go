package genetics

import (
	"testing"
	"strings"
	"github.com/yaricom/goNEAT/neat/network"
	"fmt"
	"os"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/utils"
)

func TestPlainGenomeReader_Read(t *testing.T) {
	r, err := NewGenomeReader(strings.NewReader(gnome_str), PlainGenomeEncoding)
	gnome, err := r.Read()
	if err != nil {
		t.Error(err)
		return
	}
	if gnome == nil {
		t.Error("gnome == nil")
		return
	}

	if len(gnome.Traits) != 3 {
		t.Error("len(gnome.Traits) != 3", len(gnome.Traits))
		return
	}
	ids := []int{1, 3, 2}
	for i, tr := range gnome.Traits {
		if tr.Id != ids[i] {
			t.Error("Wrong Traint ID", tr.Id, i)
		}
		if len(tr.Params) != 8 {
			t.Error("Wrong Trait's parameters lenght", len(tr.Params))
		}
		if tr.Params[0] != float64(ids[i]) / 10.0 {
			t.Error("Wrong Trait params read", tr.Params[0], i)
		}
	}

	if len(gnome.Nodes) != 4 {
		t.Error("len(gnome.Nodes) != 4", len(gnome.Nodes))
		return
	}
	for i, n := range gnome.Nodes {
		if n.Id != i + 1 {
			t.Error("Wrong NNode Id", n.Id)
		}
		if i < 3 && !n.IsSensor() {
			t.Error("Wrong NNode type, SENSOR: ", n.IsSensor())
		}

		if i == 3 {
			if !n.IsNeuron() {
				t.Error("Wrong NNode type, NEURON: ", n.IsNeuron())
			}
			if n.NeuronType != network.OutputNeuron {
				t.Error("Wrong NNode placement", n.NeuronType)
			}
		}

		if (i < 2 && n.NeuronType != network.InputNeuron) ||
			(i == 2 && n.NeuronType != network.BiasNeuron) {
			t.Error("Wrong NNode placement", n.NeuronType)
		}

	}

	if len(gnome.Genes) != 3 {
		t.Error("len(gnome.Genes) != 3", len(gnome.Genes))
	}

	for i, g := range gnome.Genes {
		if g.Link.Trait.Id != i + 1 {
			t.Error("Gene Link Traid Id is wrong", g.Link.Trait.Id)
		}
		if g.Link.InNode.Id != i + 1 {
			t.Error("Gene link's input node Id is wrong", g.Link.InNode.Id)
		}
		if g.Link.OutNode.Id != 4 {
			t.Error("Gene link's output node Id is wrong", g.Link.OutNode.Id)
		}
		if g.Link.Weight != float64(i) + 1.5 {
			t.Error("Gene link's weight is wrong", g.Link.Weight)
		}
		if g.Link.IsRecurrent {
			t.Error("Gene link's recurrent flag is wrong")
		}
		if g.InnovationNum != int64(i + 1) {
			t.Error("Gene's innovation number is wrong", g.InnovationNum)
		}
		if g.MutationNum != float64(0) {
			t.Error("Gene's mutation number is wrong", g.MutationNum)
		}
		if !g.IsEnabled {
			t.Error("Gene's enabled flag is wrong", g.IsEnabled)
		}
	}
}

func TestReadGene_readPlainTrait(t *testing.T) {
	params := []float64{
		0.40227575878298616, 0.0, 0.0, 0.0, 0.0, 0.3245553261200018, 0.0, 0.12248956525856575,
	}
	trait_id := 2
	trait_str := fmt.Sprintf("%d %g %g %g %g %g %g %g %g",
		trait_id, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
	trait, err := readPlainTrait(strings.NewReader(trait_str))
	if err != nil {
		t.Error(err)
		return
	}
	if trait.Id != trait_id {
		t.Error("trait.TraitId", trait_id, trait.Id)
	}
	for i, p := range params {
		if trait.Params[i] != p {
			t.Error("trait.Params[i] != p", trait.Params[i], p)
		}
	}
}

// Tests how NNode read working
func TestReadGene_ReadPlainNNode(t *testing.T) {
	node_id, trait_id, ntype, gen_node_label := 1, 10, network.SensorNode, network.InputNeuron
	node_str := fmt.Sprintf("%d %d %d %d", node_id, trait_id, ntype, gen_node_label)

	trait := neat.NewTrait()
	trait.Id = 10
	traits := []*neat.Trait{trait}

	node, err := readPlainNetworkNode(strings.NewReader(node_str), traits)
	if err != nil {
		t.Error(err)
		return
	}

	if node.Id != node_id {
		t.Errorf("Found node ID is not what expected, %d != %d", node_id, node.Id)
	}
	if node.Trait != trait {
		t.Error("The wrong Trait found in the node")
	}
	if node.NodeType() != ntype {
		t.Errorf("Wrong node type found, %d != %d", ntype, node.NodeType())
	}
	if node.NeuronType != gen_node_label {
		t.Errorf("The wrong node placement label found, %d != %d", gen_node_label, node.NeuronType)
	}
}

// Tests Gene ReadGene
func TestReadGene_ReadPlainGene(t *testing.T) {
	// gene  1 1 4 1.1983046913458986 0 1.0 1.1983046913458986 0
	traitId, inNodeId, outNodeId, innov_num := 1, 1, 4, int64(1)
	weight, mut_num := 1.1983046913458986, 1.1983046913458986
	recurrent, enabled := false, false
	gene_str := fmt.Sprintf("%d %d %d %g %t %d %g %t",
		traitId, inNodeId, outNodeId, weight, recurrent, innov_num, mut_num, enabled)

	trait := neat.NewTrait()
	trait.Id = 1
	nodes := []*network.NNode{
		network.NewNNode(1, network.InputNeuron),
		network.NewNNode(4, network.HiddenNeuron),
	}

	gene, err := readPlainConnectionGene(strings.NewReader(gene_str), []*neat.Trait{trait}, nodes)
	if err != nil {
		t.Error(err)
		return
	}

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
	if link.Trait.Id != traitId {
		t.Error("link.LinkTrait.TraitId", traitId, link.Trait.Id)
	}
	if link.InNode.Id != inNodeId {
		t.Error("link.InNode.NodeId", inNodeId, link.InNode.Id)
	}
	if link.OutNode.Id != outNodeId {
		t.Error("link.OutNode.NodeId", outNodeId, link.OutNode.Id)
	}
	if link.Weight != weight {
		t.Error("link.Weight", weight, link.Weight)
	}
	if link.IsRecurrent != recurrent {
		t.Error("link.IsRecurrent", recurrent, link.IsRecurrent)
	}
}

func TestPlainGenomeReader_ReadFile(t *testing.T) {
	genomePath := "../../data/xorstartgenes"
	genomeFile, err := os.Open(genomePath)
	if err != nil {
		t.Error("Failed to open genome file")
		return
	}
	r, err := NewGenomeReader(genomeFile, PlainGenomeEncoding)
	genome, err := r.Read()
	if err != nil {
		t.Error(err)
		return
	}
	if genome == nil {
		t.Error("genome == nil")
		return
	}

	if len(genome.Genes) != 3 {
		t.Error("len(gnome.Genes) != 3", len(genome.Genes))
	}
	if len(genome.Nodes) != 4 {
		t.Error("len(gnome.Nodes) != 4", len(genome.Nodes))
		return
	}
	for i, n := range genome.Nodes {
		if n.Id != i + 1 {
			t.Error("Wrong NNode Id", n.Id)
		}
		if i < 3 && !n.IsSensor() {
			t.Error("Wrong NNode type, SENSOR: ", n.IsSensor())
		}

		if i == 3 {
			if !n.IsNeuron() {
				t.Error("Wrong NNode type, NEURON: ", n.IsNeuron())
			}
			if n.NeuronType != network.OutputNeuron {
				t.Error("Wrong NNode placement", n.NeuronType)
			}
		}

		if (i == 0 && n.NeuronType != network.BiasNeuron) ||
			(i > 0 && i < 3 && n.NeuronType != network.InputNeuron) {
			t.Error("Wrong NNode placement", n.NeuronType)
		}

	}

	if len(genome.Traits) != 3 {
		t.Error("len(gnome.Traits) != 3", len(genome.Traits))
		return
	}
	for i, tr := range genome.Traits {
		if tr.Id != i + 1 {
			t.Error("Wrong Traint ID", tr.Id)
		}
		if len(tr.Params) != 8 {
			t.Error("Wrong Trait's parameters lenght", len(tr.Params))
		}
		if tr.Params[0] != float64(i + 1) / 10.0 {
			t.Error("Wrong Trait params read", tr.Params[0])
		}
	}
}

func TestYAMLGenomeReader_Read(t *testing.T) {
	genomePath := "../../data/test_seed_genome.yml"
	genomeFile, err := os.Open(genomePath)
	if err != nil {
		t.Error("Failed to open genome file")
		return
	}
	r, err := NewGenomeReader(genomeFile, YAMLGenomeEncoding)
	genome, err := r.Read()
	if err != nil {
		t.Error(err)
		return
	}
	if genome == nil {
		t.Error("genome == nil")
		return
	}

	// Test nodes
	if len(genome.Nodes) != 14 {
		t.Error("len(genome.Nodes) != 14", len(genome.Nodes))
	}
	nodes := []*network.NNode{
		{Id:1, NeuronType: network.BiasNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},

		{Id:2, NeuronType: network.InputNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:3, NeuronType: network.InputNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:4, NeuronType: network.InputNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:5, NeuronType: network.InputNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},

		{Id:6, NeuronType: network.OutputNeuron, ActivationType: utils.SigmoidBipolarActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:7, NeuronType: network.OutputNeuron, ActivationType: utils.SigmoidBipolarActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},

		{Id:8, NeuronType: network.HiddenNeuron, ActivationType: utils.LinearActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:9, NeuronType: network.HiddenNeuron, ActivationType: utils.LinearActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:10, NeuronType: network.HiddenNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},

		{Id:11, NeuronType: network.HiddenNeuron, ActivationType: utils.LinearActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:12, NeuronType: network.HiddenNeuron, ActivationType: utils.LinearActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:13, NeuronType: network.HiddenNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},

		{Id:14, NeuronType: network.HiddenNeuron, ActivationType: utils.SignActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
	}
	for i, n := range nodes {
		if n.Id != genome.Nodes[i].Id {
			t.Error("n.Id != genome.Nodes[i].Id at", i)
		}
		if n.ActivationType != genome.Nodes[i].ActivationType {
			t.Error("n.ActivationType != genome.Nodes[i].ActivationType at", i)
		}
		if n.NeuronType != genome.Nodes[i].NeuronType {
			t.Error("n.NeuronType != genome.Nodes[i].NeuronType at", i)
		}
	}

	// Check traits
	if len(genome.Traits) != 15 {
		t.Error("len(genome.Traits) != 15", len(genome.Traits))
	}
	for i, tr := range genome.Traits {
		if tr.Id != i + 1 {
			t.Error("tr.Id != i + 1 at", i)
		}
		if tr.Params[0] != float64(i + 1) / 10.0 {
			t.Error("Wrong trait param at", i, tr.Params[0])
		}
	}

	// Check Genes
	if len(genome.Genes) != 15 {
		t.Error("len(genome.Genes) != 15", len(genome.Genes))
	}
	genes := []*Gene{
		newGene(network.NewLinkWithTrait(genome.Traits[1], 0.0, nodes[0], nodes[5], false), 1, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[2], 0.0, nodes[1], nodes[5], false), 2, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[4], nodes[5], false), 3, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[4], 1.0, nodes[5], nodes[13], false), 4, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[4], 1.0, nodes[13], nodes[7], false), 5, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[1], nodes[8], false), 6, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[9], nodes[5], false), 7, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[3], nodes[11], false), 8, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 1.0, nodes[13], nodes[10], false), 9, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[12], nodes[5], false), 10, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[0], nodes[6], false), 11, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[2], nodes[6], false), 12, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[4], nodes[6], false), 13, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[9], nodes[6], false), 14, 0, true),
		newGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[12], nodes[6], false), 15, 0, true),
	}

	for i, g := range genes {
		if g.InnovationNum != genome.Genes[i].InnovationNum {
			t.Error("g.InnovationNum != genome.Genes[i].InnovationNum at:", i)
		}
		if g.IsEnabled != genome.Genes[i].IsEnabled {
			t.Error("g.IsEnabled != genome.Genes[i].IsEnabled at:", i)
		}
		if g.MutationNum != genome.Genes[i].MutationNum {
			t.Error("g.MutationNum != genome.Genes[i].MutationNum at:", i)
		}
		if g.Link.Weight != genome.Genes[i].Link.Weight {
			t.Error("g.Link.Weight != genome.Genes[i].Link.Weight at:", i)
		}
		if g.Link.InNode.Id != genome.Genes[i].Link.InNode.Id {
			t.Error("g.Link.InNode.Id != genome.Genes[i].Link.InNode.Id at:", i)
		}
		if g.Link.OutNode.Id != genome.Genes[i].Link.OutNode.Id {
			t.Error("g.Link.OutNode.Id != genome.Genes[i].Link.OutNode.Id at:", i)
		}
	}

	// Check control genes
	if len(genome.ControlGenes) != 2 {
		t.Error("len(genome.ControlGenes) != 2", len(genome.ControlGenes))
	}
	id_count := 8
	for i, g := range genome.ControlGenes {
		if g.InnovationNum != int64(i + 16) {
			t.Error("Wrong innovation number at: ", i, g.InnovationNum)
		}
		if g.MutationNum != 0.5 + float64(i) {
			t.Error("Wrong muttanion number at:", i, g.MutationNum)
		}
		if g.IsEnabled == false {
			t.Error("Wrong enabled value for module at: ", i)
		}
		if g.ControlNode == nil {
			t.Error("g.ControlNode == nil at:", i)
			return
		}
		if g.ControlNode.Id != i + 15 {
			t.Error("Wrong control node ID at:", i, g.ControlNode.Id)
		}
		if g.ControlNode.ActivationType != utils.MultiplyModuleActivation {
			t.Error("g.ControlNode.ActivationType != network.MultiplyModuleActivation at:", i)
		}
		if g.ControlNode.NeuronType != network.HiddenNeuron {
			t.Error("g.ControlNode.NeuronType != network.HiddenNeuron at:", i)
		}
		if len(g.ControlNode.Incoming) != 2 {
			t.Error("len(g.ControlNode.Incoming) != 2 at:", i, len(g.ControlNode.Incoming))
			return
		}
		if len(g.ControlNode.Outgoing) != 1 {
			t.Error("len(g.ControlNode.Outgoing) != 1 at:", i, len(g.ControlNode.Outgoing))
			return
		}
		for j, l := range g.ControlNode.Incoming {
			if l.InNode.Id != id_count {
				t.Error("l.InNode.Id != id_count", l.InNode.Id, id_count)
			}
			if l.OutNode.Id != g.ControlNode.Id {
				t.Error("l.OutNode.Id != g.ControlNode.Id", l.OutNode.Id, g.ControlNode.Id)
			}
			if l.Weight != 1.0 {
				t.Error("l.Weight != 1.0 at: ", i, j)
			}
			id_count++
		}
		if g.ControlNode.Outgoing[0].InNode.Id != g.ControlNode.Id {
			t.Error("g.ControlNode.Outgoing[0].InNode != g.ControlNode.Id", g.ControlNode.Outgoing[0].InNode, g.ControlNode.Id)
		}
		if g.ControlNode.Outgoing[0].OutNode.Id != id_count {
			t.Error("g.ControlNode.Outgoing[0].OutNode != id_count", g.ControlNode.Outgoing[0].OutNode, id_count)
		}
		if g.ControlNode.Outgoing[0].Weight != 1.0 {
			t.Error("g.ControlNode.Outgoing[0].Weight != 1.0 at:", i)
		}
		id_count++
	}
}
