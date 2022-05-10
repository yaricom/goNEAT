package network

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v3/neat"
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

func TestLink_Attributes(t *testing.T) {
	in := NewNNode(1, InputNeuron)
	out := NewNNode(2, HiddenNeuron)
	w := 12.6

	trait := &neat.Trait{Id: 1, Params: []float64{1.1, 2.3, 3.4, 4.2, 5.5, 6.7}}
	l := NewLinkWithTrait(trait, w, in, out, false)

	attrs := l.Attributes()
	require.Len(t, attrs, 3, "wrong number of attributes")

	assert.Equal(t, "weight", attrs[0].Key)
	assert.Equal(t, fmt.Sprintf("%f", w), attrs[0].Value)

	assert.Equal(t, "recurrent", attrs[1].Key)
	assert.Equal(t, fmt.Sprintf("%v", l.IsRecurrent), attrs[1].Value)

	assert.Equal(t, "parameters", attrs[2].Key)
	expected := fmt.Sprintf("%v", trait.Params)
	assert.Equal(t, expected, attrs[2].Value)
}
