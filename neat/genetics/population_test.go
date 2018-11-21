package genetics

import (
	"testing"
	"github.com/yaricom/goNEAT/neat"
	"math/rand"
	"strings"
	"bytes"
	"bufio"
)

func TestNewPopulationRandom(t *testing.T) {
	rand.Seed(42)
	in, out, nmax := 3, 2, 5
	recurrent := false
	link_prob := 0.5
	conf := neat.NeatContext{
		CompatThreshold:0.5,
		PopSize:10,
	}
	pop, err := NewPopulationRandom(in, out, nmax, recurrent, link_prob, &conf)
	if err != nil {
		t.Error(err)
	}
	if pop == nil {
		t.Error("pop == nil")
	}
	if len(pop.Organisms) != conf.PopSize {
		t.Error("len(pop.Organisms) != size")
	}
	if pop.nextNodeId != 11 {
		t.Error("pop.currNodeId != 11")
	}
	if pop.nextInnovNum != int64(101) {
		t.Error("pop.currInnovNum != 101")
	}
	if len(pop.Species) == 0 {
		t.Error("len(pop.Species) == 0")
	}

	for _, org := range pop.Organisms {
		if len(org.Genotype.Genes) == 0 {
			t.Error("len(org.GNome.Genes) == 0")
		}
		if len(org.Genotype.Nodes) == 0 {
			t.Error("len(org.GNome.Nodes) == 0")
		}
		if len(org.Genotype.Traits) == 0 {
			t.Error("len(org.GNome.Traits) == 0")
		}
		if org.Genotype.Phenotype == nil {
			t.Error("org.GNome.Phenotype == nil")
		}
	}

}

func TestNewPopulation(t *testing.T) {
	rand.Seed(42)
	in, out, nmax, n := 3, 2, 5, 3
	recurrent := false
	link_prob := 0.5
	conf := neat.NeatContext{
		CompatThreshold:0.5,
		PopSize:10,
	}
	gen := newGenomeRand(1, in, out, n, nmax, recurrent, link_prob)

	pop, err := NewPopulation(gen, &conf)
	if err != nil {
		t.Error(err)
	}
	if pop == nil {
		t.Error("pop == nil")
	}
	if len(pop.Organisms) != conf.PopSize {
		t.Error("len(pop.Organisms) != conf.PopSize")
	}
	last_node_id, _ := gen.getLastNodeId()
	if pop.nextNodeId != int32(last_node_id + 1) {
		t.Error("pop.currNodeId != last_node_id")
	}
	last_gene_innov_num, _ := gen.getNextGeneInnovNum()
	if pop.nextInnovNum != last_gene_innov_num {
		t.Error("pop.currInnovNum != last_gene_innov_num")
	}

	for _, org := range pop.Organisms {
		if len(org.Genotype.Genes) == 0 {
			t.Error("len(org.GNome.Genes) == 0")
		}
		if len(org.Genotype.Nodes) == 0 {
			t.Error("len(org.GNome.Nodes) == 0")
		}
		if len(org.Genotype.Traits) == 0 {
			t.Error("len(org.GNome.Traits) == 0")
		}
		if org.Genotype.Phenotype == nil {
			t.Error("org.GNome.Phenotype == nil")
		}
	}
}

func TestReadPopulation(t *testing.T) {
	pop_str := "genomestart 1\n" +
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
		"genomeend 1\n" +
		"genomestart 2\n" +
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
		"genomeend 2\n"
	conf := neat.NeatContext{
		CompatThreshold:0.5,
	}
	pop, err := ReadPopulation(strings.NewReader(pop_str), &conf)
	if err != nil {
		t.Error(err)
		return
	}
	if pop == nil {
		t.Error("pop == nil")
	}
	if len(pop.Organisms) != 2 {
		t.Error("len(pop.Organisms) != size")
	}
	if len(pop.Species) != 1 {
		// because genomes are identical
		t.Error("len(pop.Species) != 1", len(pop.Species))
	}
}

func TestPopulation_verify(t *testing.T) {
	// first create population
	pop_str := "genomestart 1\n" +
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
		"genomeend 1\n" +
		"genomestart 2\n" +
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
		"genomeend 2\n"
	conf := neat.NeatContext{
		CompatThreshold:0.5,
	}
	pop, err := ReadPopulation(strings.NewReader(pop_str), &conf)
	if err != nil {
		t.Error(err)
		return
	}
	if pop == nil {
		t.Error("pop == nil")
	}

	// then verify created
	res, err := pop.Verify()
	if err != nil {
		t.Error(err)
	}
	if !res {
		t.Error("Population verification failed, but must not")
	}
}

func TestPopulation_Write(t *testing.T) {
	// first create population
	pop_str := "genomestart 1\n" +
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
		"genomeend 1\n" +
		"genomestart 2\n" +
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
		"genomeend 2\n"
	conf := neat.NeatContext{
		CompatThreshold:0.5,
	}
	pop, err := ReadPopulation(strings.NewReader(pop_str), &conf)
	if err != nil {
		t.Error(err)
		return
	}
	if pop == nil {
		t.Error("pop == nil")
		return
	}

	// write it again and test
	out_buf := bytes.NewBufferString("")
	pop.Write(out_buf)

	_, p_str_r, err_p := bufio.ScanLines([]byte(pop_str), true)
	_, o_str_r, err_o := bufio.ScanLines(out_buf.Bytes(), true)
	if err_p != nil || err_o != nil {
		t.Error("Failed to parse strings", err_o, err_p)
	}
	for i, gsr := range p_str_r {
		if gsr != o_str_r[i] {
			t.Error("Lines mismatch", gsr, o_str_r[i])
		}
	}
}
