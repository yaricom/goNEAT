package genetics

import (
	"testing"
	"strings"
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat"
	"bytes"
	"bufio"
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
		if g.Link.LinkTrait.Id != i + 1 {
			t.Error("Gene Link Traid Id is wrong", g.Link.LinkTrait.Id)
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

// Test write Genome
func TestGenome_WriteGenome(t *testing.T) {
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

	gnome := NewGenome(1, traits, nodes, genes)
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
