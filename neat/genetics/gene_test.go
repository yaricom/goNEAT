package genetics

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat"
	"reflect"
	"github.com/yaricom/goNEAT/neat/utils"
)

// Tests Gene WriteGene
func TestNewGeneCopy(t *testing.T) {
	nodes := []*network.NNode{
		{Id:1, NeuronType: network.InputNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:2, NeuronType: network.OutputNeuron, ActivationType: utils.SigmoidSteepenedActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
	}
	trait := &neat.Trait{Id:1, Params:[]float64{0.1, 0, 0, 0, 0, 0, 0, 0}}
	g1 := NewGeneWithTrait(trait, 3.2, nodes[0], nodes[1], true, 42, 5.2)

	// test
	g := NewGeneCopy(g1, trait, nodes[0], nodes[1])
	if g.Link.InNode.Id != nodes[0].Id {
		t.Error("g.Link.InNode.Id != nodes[0].Id", g.Link.InNode.Id)
	}
	if g.Link.OutNode.Id != nodes[1].Id {
		t.Error("g.Link.OutNode.Id != nodes[1].Id", g.Link.OutNode.Id)
	}
	if g.Link.Trait.Id != trait.Id {
		t.Error("g.Link.Trait.Id != trait.Id", g.Link.Trait.Id)
	}
	if reflect.DeepEqual(g.Link.Trait.Params, trait.Params) == false {
		t.Error("reflect.DeepEqual(g.Link.Trait.Params, trait.Params) == false")
	}
	if g.InnovationNum != g1.InnovationNum {
		t.Error("g.InnovationNum != g1.InnovationNum", g.InnovationNum, g1.InnovationNum)
	}
	if g.MutationNum != g1.MutationNum {
		t.Error("g.MutationNum != g1.MutationNum", g.MutationNum, g1.MutationNum)
	}
	if g.IsEnabled != g1.IsEnabled {
		t.Error("g.IsEnabled != g1.IsEnabled", g.IsEnabled, g1.IsEnabled)
	}
}
