package network

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestLink_From(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)

	l := NewLink(1.0, in, out, false)

	from := l.From()
	assert.Equal(t, in, from)
}

func TestLink_To(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)

	l := NewLink(1.0, in, out, false)

	to := l.To()
	assert.Equal(t, out, to)
}

func TestLink_Weight(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)

	w := 12.3
	l := NewLink(w, in, out, false)

	res := l.Weight()
	assert.Equal(t, w, res)
}

func TestLink_ReversedEdge(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)

	l := NewLink(1.0, in, out, false)

	reversed := l.ReversedEdge()
	assert.Equal(t, l, reversed)
}
