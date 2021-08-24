package network

import (
	"github.com/stretchr/testify/assert"
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
