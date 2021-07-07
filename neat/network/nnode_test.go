package network

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

// Tests NNode SensorLoad
func TestNNode_SensorLoad(t *testing.T) {
	node := NewNNode(1, InputNeuron)

	load := 21.0
	res := node.SensorLoad(load)
	require.True(t, res, "Failed to SensorLoad")
	assert.EqualValues(t, 1, node.ActivationsCount)
	assert.Equal(t, load, node.Activation)
	assert.Equal(t, load, node.GetActiveOut())

	load2 := 36.0
	res = node.SensorLoad(load2)
	require.True(t, res, "Failed to SensorLoad")
	assert.EqualValues(t, 2, node.ActivationsCount)
	assert.Equal(t, load2, node.Activation)
	// Check activation and time delayed activation
	assert.Equal(t, load2, node.GetActiveOut())
	assert.Equal(t, load, node.GetActiveOutTd())

	// Check loading of incorrect type node
	//
	nodeN := NewNNode(1, HiddenNeuron)
	res = nodeN.SensorLoad(load)
	assert.False(t, res, "Non SENSOR node can not be loaded")
}

// Tests NNode AddIncoming
func TestNNode_AddIncoming(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)

	weight := 1.5
	node2.addIncoming(node, weight)
	assert.Len(t, node2.Incoming, 1, "Wrong number of incoming nodes")

	link := node2.Incoming[0]
	assert.Equal(t, weight, link.ConnectionWeight, "Wrong incoming link weight")
	assert.Equal(t, node, link.InNode, "Wrong InNode in Link")
	assert.Equal(t, node2, link.OutNode, "Wrong OutNode in Link")
}

// Tests NNode Depth
func TestNNode_Depth(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)
	node3 := NewNNode(3, OutputNeuron)

	node2.addIncoming(node, 15.0)
	node3.addIncoming(node2, 20.0)

	depth, err := node3.Depth(0)
	require.NoError(t, err)
	assert.Equal(t, 2, depth)
}

func TestNNode_DepthWithLoop(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)
	node3 := NewNNode(3, OutputNeuron)

	node2.addIncoming(node, 15.0)
	node3.addIncoming(node2, 20.0)
	node2.addIncoming(node3, 10.0)
	depth, err := node3.Depth(0)
	require.NoError(t, err)
	assert.Equal(t, 2, depth)
}

// Tests NNode Flushback
func TestNNode_Flushback(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	load := 34.0
	load2 := 14.0
	node.SensorLoad(load)
	node.SensorLoad(load2)

	// check that node state has been updated
	assert.EqualValues(t, 2, node.ActivationsCount)
	assert.Equal(t, 14.0, node.Activation)

	// Check activation and time delayed activation
	assert.Equal(t, load2, node.GetActiveOut())
	assert.Equal(t, load, node.GetActiveOutTd())

	// check flush back
	//
	node.Flushback()

	assert.Zero(t, node.ActivationsCount)
	assert.Zero(t, node.Activation)

	// Check activation and time delayed activation
	assert.Zero(t, node.GetActiveOut())
	assert.Zero(t, node.GetActiveOutTd())
}
