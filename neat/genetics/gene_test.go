package genetics

import (
	"github.com/stretchr/testify/assert"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/math"
	"github.com/yaricom/goNEAT/v4/neat/network"
	"testing"
)

// Tests Gene WriteGene
func TestNewGeneCopy(t *testing.T) {
	nodes := []*network.NNode{
		{Id: 1, NeuronType: network.InputNeuron, ActivationType: math.NullActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 2, NeuronType: network.OutputNeuron, ActivationType: math.SigmoidSteepenedActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
	}
	trait := &neat.Trait{Id: 1, Params: []float64{0.1, 0, 0, 0, 0, 0, 0, 0}}
	g1 := NewGeneWithTrait(trait, 3.2, nodes[0], nodes[1], true, 42, 5.2)

	// test
	g := NewGeneCopy(g1, trait, nodes[0], nodes[1])
	assert.Equal(t, nodes[0].Id, g.Link.InNode.Id)
	assert.Equal(t, nodes[1].Id, g.Link.OutNode.Id)
	assert.Equal(t, trait.Id, g.Link.Trait.Id)
	assert.Equal(t, trait.Params, g.Link.Trait.Params)
	assert.Equal(t, g1.InnovationNum, g.InnovationNum)
	assert.Equal(t, g1.InnovationNum, g.InnovationNum)
	assert.Equal(t, g1.MutationNum, g.MutationNum)
	assert.Equal(t, g1.IsEnabled, g.IsEnabled)
}
