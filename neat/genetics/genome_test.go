package genetics

import (
	"testing"
	"strings"
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat"
	"bytes"
	"bufio"
	"reflect"
)

const gnome_str = "genomestart 1\n" +
	"trait 1 0.1 0 0 0 0 0 0 0\n" +
	"trait 2 0.2 0 0 0 0 0 0 0\n" +
	"trait 3 0.3 0 0 0 0 0 0 0\n" +
	"node 1 0 1 1\n" +
	"node 2 0 1 1\n" +
	"node 3 0 1 3\n" +
	"node 4 0 0 2\n" +
	"gene 1 1 4 1.5 false 1 0 true\n" +
	"gene 2 2 4 2.5 false 2 0 true\n" +
	"gene 3 3 4 3.5 false 3 0 true\n" +
	"genomeend 1"

func buildTestGenome(id int) *Genome {
	traits := []*neat.Trait {
		neat.ReadTrait(strings.NewReader("1 0.1 0 0 0 0 0 0 0")),
		neat.ReadTrait(strings.NewReader("2 0.2 0 0 0 0 0 0 0")),
		neat.ReadTrait(strings.NewReader("3 0.3 0 0 0 0 0 0 0")),
	}

	nodes := []*network.NNode {
		network.ReadNNode(strings.NewReader("1 0 1 1"), traits),
		network.ReadNNode(strings.NewReader("2 0 1 1"), traits),
		network.ReadNNode(strings.NewReader("3 0 1 3"), traits),
		network.ReadNNode(strings.NewReader("4 0 0 2"), traits),
	}

	genes := []*Gene {
		ReadGene(strings.NewReader("1 1 4 1.5 false 1 0 true"), traits, nodes),
		ReadGene(strings.NewReader("2 2 4 2.5 false 2 0 true"), traits, nodes),
		ReadGene(strings.NewReader("3 3 4 3.5 false 3 0 true"), traits, nodes),
	}

	return NewGenome(id, traits, nodes, genes)
}

// Tests Genome reading
func TestGenome_ReadGenome(t *testing.T) {
	gnome, err := ReadGenome(strings.NewReader(gnome_str), 2)
	if gnome != nil {
		t.Error("Genome read should fail due ID mismatch")
	}
	if err == nil {
		t.Error("Genome read should fail due ID mismatch")
	}

	gnome, err = ReadGenome(strings.NewReader(gnome_str), 1)
	if err != nil {
		t.Error("err != nil", err)
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
		if i < 3 && n.NType != network.SENSOR {
			t.Error("Wrong NNode type", n.NType)
		}

		if i == 3 {
			if n.NType != network.NEURON {
				t.Error("Wrong NNode type", n.NType)
			}
			if n.GenNodeLabel != network.OUTPUT {
				t.Error("Wrong NNode placement", n.GenNodeLabel)
			}
		}

		if (i < 2 && n.GenNodeLabel != network.INPUT) ||
			(i == 2 && n.GenNodeLabel != network.BIAS) {
			t.Error("Wrong NNode placement", n.GenNodeLabel)
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
		if g.InnovationNum != i + 1 {
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

// Test write Genome
func TestGenome_WriteGenome(t *testing.T) {

	gnome := buildTestGenome(1)
	out_buf := bytes.NewBufferString("")
	gnome.WriteGenome(out_buf)

	_, g_str_r, err_g := bufio.ScanLines([]byte(gnome_str), true)
	_, o_str_r, err_o := bufio.ScanLines(out_buf.Bytes(), true)
	if err_g != nil || err_o != nil {
		t.Error("Failed to parse strings", err_o, err_g)
	}
	for i, gsr := range g_str_r {
		if gsr != o_str_r[i] {
			t.Error("Lines mismatch", gsr, o_str_r[i])
		}
	}
}

// Test create random genome
func TestGenome_NewGenomeRand(t *testing.T) {
	new_id, in, out, n, nmax := 1, 3, 2, 2, 5
	recurrent := false
	link_prob := 0.5

	gnome := NewGenomeRand(new_id, in, out, n, nmax, recurrent, link_prob)

	if gnome == nil {
		t.Error("Failed to create random genome")
	}
	if len(gnome.Nodes) != in + n + out {
		t.Error("len(gnome.Nodes) != in + nmax + out", len(gnome.Nodes), in + n + out)
	}
	if len(gnome.Genes) < in + n + out {
		t.Error("Failed to create genes", len(gnome.Genes))
	}

	for _, g := range gnome.Genes {
		t.Log(g)
	}
}

// Test genesis
func TestGenome_Genesis(t *testing.T)  {
	gnome := buildTestGenome(1)

	net_id := 10

	net := gnome.genesis(net_id)
	if net == nil {
		t.Error("Failed to do network genesis")
	}
	if net.Id != net_id {
		t.Error("net.Id != net_id", net.Id)
	}
	if net.NodeCount() != len(gnome.Nodes) {
		t.Error("net.NodeCount() != len(nodes)", net.NodeCount(), len(gnome.Nodes))
	}
	if net.LinkCount() != len(gnome.Genes) {
		t.Error("net.LinkCount() != len(genes)", net.LinkCount(), len(gnome.Genes))
	}
}

// Test duplicate
func TestGenome_Duplicate(t *testing.T)  {
	gnome := buildTestGenome(1)

	new_gnome := gnome.duplicate(2)
	if new_gnome.Id != 2 {
		t.Error("new_gnome.Id != 2", new_gnome.Id)
	}

	if len(new_gnome.Traits) != len(gnome.Traits) {
		t.Error("len(new_gnome.Traits) != len(gnome.Traits)", len(new_gnome.Traits), len(gnome.Traits))
	}
	if len(new_gnome.Nodes) != len(gnome.Nodes) {
		t.Error("len(new_gnome.Nodes) != len(gnome.Nodes)", len(new_gnome.Nodes), len(gnome.Nodes))
	}
	if len(new_gnome.Genes) != len(gnome.Genes) {
		t.Error("len(new_gnome.Genes) != len(gnome.Genes)", len(new_gnome.Genes), len(gnome.Genes))
	}

	for i, tr := range new_gnome.Traits {
		if !reflect.DeepEqual(tr, gnome.Traits[i]) {
			t.Error("Wrong trait found in new genome")
		}
	}
	for i, nd := range new_gnome.Nodes {
		gnome.Nodes[i].Duplicate = nil
		if !reflect.DeepEqual(nd, gnome.Nodes[i]) {
			t.Error("Wrong node found in new genome", nd, gnome.Nodes[i])
		}
	}

	for i, g := range new_gnome.Genes {
		if !reflect.DeepEqual(g, gnome.Genes[i]) {
			t.Error("Wrong gene found", g, gnome.Genes[i])
		}
	}
}
