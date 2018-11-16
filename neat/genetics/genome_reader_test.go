package genetics

import (
	"testing"
	"strings"
	"github.com/yaricom/goNEAT/neat/network"
	"fmt"
	"os"
	"github.com/yaricom/goNEAT/neat"
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
	for i, tr := range gnome.Traits {
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
			t.Error("Gene's innovation number is wrong",  g.InnovationNum)
		}
		if g.MutationNum != float64(0) {
			t.Error("Gene's mutation number is wrong",  g.MutationNum)
		}
		if !g.IsEnabled {
			t.Error("Gene's enabled flag is wrong",  g.IsEnabled)
		}
	}
}

func TestReadGene_readPlainTrait(t *testing.T)  {
	params := []float64 {
		0.40227575878298616, 0.0, 0.0, 0.0, 0.0, 0.3245553261200018, 0.0, 0.12248956525856575,
	}
	trait_id := 2
	trait_str := fmt.Sprintf("%d %g %g %g %g %g %g %g %g",
		trait_id, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
	trait := readPlainTrait(strings.NewReader(trait_str))
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

	node := readPlainNetworkNode(strings.NewReader(node_str), traits)

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
func TestReadGene_ReadPlainGene(t *testing.T)  {
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

	gene := readPlainConnectionGene(strings.NewReader(gene_str), []*neat.Trait{trait}, nodes)

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
			if !n.IsNeuron()  {
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
}
