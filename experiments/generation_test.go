package experiments

import (
	"bytes"
	"encoding/gob"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat/network"
	"strings"
	"testing"
	"time"
	"reflect"
)

// Tests encoding/decoding of generation
func TestGeneration_Encode_Decode(t *testing.T) {
	genome_id, fitness := 10, 23.0
	gen := buildTestGeneration(genome_id, fitness)

	var buff bytes.Buffer
	enc := gob.NewEncoder(&buff)

	// encode generation
	err := gen.Encode(enc)
	if err != nil {
		t.Error("failed to encode generation", err)
		return
	}

	// decode generation
	data := buff.Bytes()
	dec := gob.NewDecoder(bytes.NewBuffer(data))
	dgen := &Generation{}
	err = dgen.Decode(dec)
	if err != nil {
		t.Error("failed to dencode generation", err)
		return
	}

	//  and test fields
	if gen.Id != dgen.Id {
		t.Error("gen.Id != dgen.Id")
	}
	if gen.Executed != dgen.Executed {
		t.Errorf("gen.Executed != dgen.Executed, %s != %s\n", gen.Executed, dgen.Executed)
	}
	if gen.Solved != dgen.Solved {
		t.Error("gen.Solved != dgen.Solved")
	}

	if !reflect.DeepEqual(gen.Fitness, dgen.Fitness) {
		t.Error("Fitness values mismatch")
	}
	if !reflect.DeepEqual(gen.Age, dgen.Age) {
		t.Error("Age values mismatch")
	}
	if !reflect.DeepEqual(gen.Compexity, dgen.Compexity) {
		t.Error("Compexity values mismatch")
	}

	if gen.Diversity != dgen.Diversity {
		t.Error("gen.Diversity != dgen.Diversity")
	}
	if gen.WinnerEvals != dgen.WinnerEvals {
		t.Error("gen.WinnerEvals != dgen.WinnerEvals")
	}
	if gen.WinnerNodes != dgen.WinnerNodes {
		t.Error("gen.WinnerNodes != dgen.WinnerNodes")
	}
	if gen.WinnerGenes != dgen.WinnerGenes {
		t.Error("gen.WinnerGenes != dgen.WinnerGenes")
	}

	if dgen.Best.Fitness != fitness {
		t.Error("dgen.Best.Fitness != fitness")
	}
	if dgen.Best.Genotype.Id != genome_id {
		t.Error("dgen.Best.Genotype.Id != genome_id")
	}

	for i, tr := range dgen.Best.Genotype.Traits {
		if !reflect.DeepEqual(tr, gen.Best.Genotype.Traits[i]) {
			t.Error("Wrong trait found in new genome")
		}
	}
	for i, nd := range dgen.Best.Genotype.Nodes {
		nd.Duplicate = nil
		if !reflect.DeepEqual(nd, gen.Best.Genotype.Nodes[i]) {
			t.Error("Wrong node found", nd, gen.Best.Genotype.Nodes[i])
		}
	}

	for i, g := range dgen.Best.Genotype.Genes {
		if !reflect.DeepEqual(g, gen.Best.Genotype.Genes[i]) {
			t.Error("Wrong gene found", g, gen.Best.Genotype.Genes[i])
		}
	}
}

func buildTestGeneration(gen_id int, fitness float64) *Generation {
	epoch := Generation{}
	epoch.Id = gen_id
	epoch.Executed = time.Now().Round(time.Second)
	epoch.Solved = true
	epoch.Fitness = Floats{10.0, 30.0, 40.0, fitness}
	epoch.Age = Floats{1.0, 3.0, 4.0, 10.0}
	epoch.Compexity = Floats{34.0, 21.0, 56.0, 15.0}
	epoch.Diversity = 32
	epoch.WinnerEvals = 12423
	epoch.WinnerNodes = 7
	epoch.WinnerGenes = 5

	genome := buildTestGenome(gen_id)
	org := genetics.Organism{Fitness:fitness, Genotype:genome, Generation:gen_id}
	epoch.Best = &org

	return &epoch
}

func buildTestGenome(id int) *genetics.Genome {
	traits := []*neat.Trait{
		neat.ReadTrait(strings.NewReader("1 0.1 0 0 0 0 0 0 0")),
		neat.ReadTrait(strings.NewReader("2 0.2 0 0 0 0 0 0 0")),
		neat.ReadTrait(strings.NewReader("3 0.3 0 0 0 0 0 0 0")),
	}

	nodes := []*network.NNode{
		network.ReadNNode(strings.NewReader("1 0 1 1"), traits),
		network.ReadNNode(strings.NewReader("2 0 1 1"), traits),
		network.ReadNNode(strings.NewReader("3 0 1 3"), traits),
		network.ReadNNode(strings.NewReader("4 0 0 2"), traits),
	}

	genes := []*genetics.Gene{
		genetics.ReadGene(strings.NewReader("1 1 4 1.5 false 1 0 true"), traits, nodes),
		genetics.ReadGene(strings.NewReader("2 2 4 2.5 false 2 0 true"), traits, nodes),
		genetics.ReadGene(strings.NewReader("3 3 4 3.5 false 3 0 true"), traits, nodes),
	}

	return genetics.NewGenome(id, traits, nodes, genes)
}
