package network

import (
	"github.com/stretchr/testify/assert"
	"github.com/yaricom/goNEAT/v4/neat"
	"testing"
)

func TestLink_IsEqualGenetically(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)

	link1 := NewLink(1.0, in, out, false)
	link2 := NewLink(2.0, in, out, false)

	equals := link1.IsEqualGenetically(link2)
	assert.True(t, equals)

	link2 = NewLink(2.0, in, out, true)
	equals = link1.IsEqualGenetically(link2)
	assert.False(t, equals)

	link2 = NewLink(2.0, in, NewNNode(3, HiddenNeuron), false)
	equals = link1.IsEqualGenetically(link2)
	assert.False(t, equals)

	link2 = NewLink(2.0, NewNNode(3, InputNeuron), out, false)
	equals = link1.IsEqualGenetically(link2)
	assert.False(t, equals)
}

func TestNewLinkCopy(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)

	trait := &neat.Trait{Id: 1, Params: []float64{1.1, 2.3, 3.4, 4.2, 5.5, 6.7}}
	link := NewLinkWithTrait(trait, 1.0, in, out, false)

	inCopy := NewNNode(3, InputNeuron)
	outCopy := NewNNode(4, HiddenNeuron)
	linkCopy := NewLinkCopy(link, inCopy, outCopy)

	assert.Equal(t, link.ConnectionWeight, linkCopy.ConnectionWeight, "wrong weight")
	assert.Equal(t, link.Params, linkCopy.Params, "wrong parameters")
	assert.Equal(t, link.IsRecurrent, linkCopy.IsRecurrent, "wrong recurrent")
	assert.Equal(t, inCopy, linkCopy.InNode, "wrong input node")
	assert.Equal(t, outCopy, linkCopy.OutNode, "wrong output node")
}

func TestNewLinkWithTrait(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)
	w := 10.9

	trait := &neat.Trait{Id: 1, Params: []float64{1.1, 2.3, 3.4, 4.2, 5.5, 6.7}}
	link := NewLinkWithTrait(trait, w, in, out, false)

	assert.Equal(t, in, link.InNode)
	assert.Equal(t, out, link.OutNode)
	assert.Equal(t, trait.Params, link.Params)
	assert.Equal(t, w, link.ConnectionWeight)
	assert.False(t, link.IsRecurrent)
}

func TestNewLink(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)
	w := 10.9

	link := NewLink(w, in, out, true)
	assert.Equal(t, in, link.InNode)
	assert.Equal(t, out, link.OutNode)
	assert.Equal(t, w, link.ConnectionWeight)
	assert.True(t, link.IsRecurrent)
}

func TestLink_String(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)
	w := 10.9

	link := NewLink(w, in, out, true)

	str := link.String()
	assert.NotEmpty(t, str)
}

func TestLink_IDString(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)

	link := NewLink(0, in, out, true)
	idStr := link.IDString()
	assert.NotEmpty(t, idStr)

	expectedIdStr := "1-2"
	assert.Equal(t, expectedIdStr, idStr)
}
