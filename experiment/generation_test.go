package experiment

import (
	"bytes"
	"encoding/gob"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"testing"
	"time"
)

// Tests encoding/decoding of generation
func TestGeneration_Encode_Decode(t *testing.T) {
	genomeId, fitness := 10, 23.0
	gen := buildTestGeneration(genomeId, fitness)

	var buff bytes.Buffer
	enc := gob.NewEncoder(&buff)

	// encode generation
	err := gen.Encode(enc)
	require.NoError(t, err, "failed to encode generation")

	// decode generation
	data := buff.Bytes()
	dec := gob.NewDecoder(bytes.NewBuffer(data))
	dgen := &Generation{}
	err = dgen.Decode(dec)
	require.NoError(t, err, "failed to decode generation")

	//  and test fields
	assert.EqualValues(t, gen, dgen)
}

func buildTestGeneration(genId int, fitness float64) *Generation {
	epoch := Generation{}
	epoch.Id = genId
	epoch.Executed = time.Now().Round(time.Second)
	epoch.Solved = true
	epoch.Fitness = Floats{10.0, 30.0, 40.0, fitness}
	epoch.Age = Floats{1.0, 3.0, 4.0, 10.0}
	epoch.Complexity = Floats{34.0, 21.0, 56.0, 15.0}
	epoch.Diversity = 32
	epoch.WinnerEvals = 12423
	epoch.WinnerNodes = 7
	epoch.WinnerGenes = 5

	genome := buildTestGenome(genId)
	org := genetics.Organism{Fitness: fitness, Genotype: genome, Generation: genId}
	epoch.Best = &org

	return &epoch
}

func buildTestGenome(id int) *genetics.Genome {
	traits := []*neat.Trait{
		{Id: 1, Params: []float64{0.1, 0, 0, 0, 0, 0, 0, 0}},
		{Id: 3, Params: []float64{0.3, 0, 0, 0, 0, 0, 0, 0}},
		{Id: 2, Params: []float64{0.2, 0, 0, 0, 0, 0, 0, 0}},
	}

	nodes := []*network.NNode{
		{Id: 1, NeuronType: network.InputNeuron, ActivationType: math.NullActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 2, NeuronType: network.InputNeuron, ActivationType: math.NullActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 3, NeuronType: network.BiasNeuron, ActivationType: math.SigmoidSteepenedActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 4, NeuronType: network.OutputNeuron, ActivationType: math.SigmoidSteepenedActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
	}

	genes := []*genetics.Gene{
		genetics.NewGeneWithTrait(traits[0], 1.5, nodes[0], nodes[3], false, 1, 0),
		genetics.NewGeneWithTrait(traits[2], 2.5, nodes[1], nodes[3], false, 2, 0),
		genetics.NewGeneWithTrait(traits[1], 3.5, nodes[2], nodes[3], false, 3, 0),
	}

	return genetics.NewGenome(id, traits, nodes, genes)
}
