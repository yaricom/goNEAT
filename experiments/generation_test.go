package experiments

import (
	"bytes"
	"encoding/gob"
	"testing"
	"time"
	"reflect"

	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/utils"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat/network"
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
	deepCompareGenerations(gen, dgen, t)
}

func deepCompareGenerations(first, second *Generation, t *testing.T) {
	if first.Id != second.Id {
		t.Error("first.Id != second.Id")
	}
	if first.Executed != second.Executed {
		t.Errorf("first.Executed != second.Executed, %s != %s\n", first.Executed, second.Executed)
	}
	if first.Solved != second.Solved {
		t.Error("first.Solved != second.Solved")
	}

	if !reflect.DeepEqual(first.Fitness, second.Fitness) {
		t.Error("Fitness values mismatch")
	}
	if !reflect.DeepEqual(first.Age, second.Age) {
		t.Error("Age values mismatch")
	}
	if !reflect.DeepEqual(first.Compexity, second.Compexity) {
		t.Error("Compexity values mismatch")
	}

	if first.Diversity != second.Diversity {
		t.Error("first.Diversity != second.Diversity")
	}
	if first.WinnerEvals != second.WinnerEvals {
		t.Error("first.WinnerEvals != second.WinnerEvals")
	}
	if first.WinnerNodes != second.WinnerNodes {
		t.Error("first.WinnerNodes != second.WinnerNodes ")
	}
	if first.WinnerGenes != second.WinnerGenes {
		t.Error("first.WinnerGenes != second.WinnerGenes")
	}

	if first.Best.Fitness != second.Best.Fitness {
		t.Error("first.Best.Fitness != second.Best.Fitness")
	}
	if first.Best.Genotype.Id != second.Best.Genotype.Id {
		t.Error("first.Best.Genotype.Id != second.Best.Genotype.Id")
	}

	for i, tr := range second.Best.Genotype.Traits {
		if !reflect.DeepEqual(tr, first.Best.Genotype.Traits[i]) {
			t.Error("Wrong trait found in new genome")
		}
	}
	for i, nd := range second.Best.Genotype.Nodes {
		if !reflect.DeepEqual(nd, first.Best.Genotype.Nodes[i]) {
			t.Error("Wrong node found", nd, first.Best.Genotype.Nodes[i])
		}
	}

	for i, g := range second.Best.Genotype.Genes {
		if !reflect.DeepEqual(g, first.Best.Genotype.Genes[i]) {
			t.Error("Wrong gene found", g, first.Best.Genotype.Genes[i])
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
		{Id:1, Params:[]float64{0.1, 0, 0, 0, 0, 0, 0, 0}},
		{Id:3, Params:[]float64{0.3, 0, 0, 0, 0, 0, 0, 0}},
		{Id:2, Params:[]float64{0.2, 0, 0, 0, 0, 0, 0, 0}},
	}

	nodes := []*network.NNode{
		{Id:1, NeuronType: network.InputNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:2, NeuronType: network.InputNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:3, NeuronType: network.BiasNeuron, ActivationType: utils.SigmoidSteepenedActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:4, NeuronType: network.OutputNeuron, ActivationType: utils.SigmoidSteepenedActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
	}

	genes := []*genetics.Gene{
		genetics.NewGeneWithTrait(traits[0], 1.5, nodes[0], nodes[3], false, 1, 0),
		genetics.NewGeneWithTrait(traits[2], 2.5, nodes[1], nodes[3], false, 2, 0),
		genetics.NewGeneWithTrait(traits[1], 3.5, nodes[2], nodes[3], false, 3, 0),
	}

	return genetics.NewGenome(id, traits, nodes, genes)
}
