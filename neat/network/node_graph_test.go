package network

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestNNode_Attributes(t *testing.T) {
	params := []float64{1.1, 2.3, 3.4, 4.2, 5.5, 6.7}
	node := NewNNode(1, InputNeuron)

	attrs := node.Attributes()
	require.Len(t, attrs, 2, "wrong attributes length")
	assert.Equal(t, "neuron_type", attrs[0].Key)
	assert.Equal(t, "INPT", attrs[0].Value)

	assert.Equal(t, "activation_type", attrs[1].Key)
	assert.Equal(t, "SigmoidSteepenedActivation", attrs[1].Value)

	node.Params = params
	attrs = node.Attributes()
	require.Len(t, attrs, 3, "wrong attributes length")
	assert.Equal(t, "parameters", attrs[2].Key)
	assert.Equal(t, fmt.Sprintf("%v", params), attrs[2].Value)
}

func TestNNode_ID(t *testing.T) {
	node := NewNNode(101, InputNeuron)

	assert.EqualValues(t, 101, node.ID())
}
