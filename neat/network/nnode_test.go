package network

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"testing"
)

func TestNewNNode(t *testing.T) {
	node := NewNNode(1, InputNeuron)

	require.NotNil(t, node)
	assert.Equal(t, 1, node.Id)
	assert.Equal(t, math.SigmoidSteepenedActivation, node.ActivationType)
	assert.Equal(t, InputNeuron, node.NeuronType)
	assert.NotNil(t, node.Incoming)
	assert.NotNil(t, node.Outgoing)
}

func TestNewNNodeCopy(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	trait := &neat.Trait{Id: 1, Params: []float64{1.1, 2.3, 3.4, 4.2, 5.5, 6.7}}

	nodeCopy := NewNNodeCopy(node, trait)

	require.NotNil(t, nodeCopy)
	assert.Equal(t, node.Id, nodeCopy.Id)
	assert.Equal(t, node.ActivationType, nodeCopy.ActivationType)
	assert.Equal(t, node.NeuronType, nodeCopy.NeuronType)
	assert.Equal(t, trait, nodeCopy.Trait)
	assert.NotNil(t, node.Incoming)
	assert.NotNil(t, node.Outgoing)
}

func TestNewNetworkNode(t *testing.T) {
	node := NewNetworkNode()

	require.NotNil(t, node)
	assert.Equal(t, math.SigmoidSteepenedActivation, node.ActivationType)
	assert.Equal(t, HiddenNeuron, node.NeuronType)
	assert.NotNil(t, node.Incoming)
	assert.NotNil(t, node.Outgoing)
}

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
	node2.AddIncoming(node, weight)
	assert.Len(t, node2.Incoming, 1, "Wrong number of incoming nodes")
	assert.Len(t, node.Outgoing, 0)

	link := node2.Incoming[0]
	assert.Equal(t, weight, link.ConnectionWeight, "Wrong incoming link weight")
	assert.Equal(t, node, link.InNode, "Wrong InNode in Link")
	assert.Equal(t, node2, link.OutNode, "Wrong OutNode in Link")
}

func TestNNode_AddOutgoing(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)

	weight := 1.5
	node.AddOutgoing(node2, weight)
	assert.Len(t, node.Outgoing, 1, "wrong number of outgoing nodes")
	assert.Len(t, node2.Incoming, 0)

	link := node.Outgoing[0]
	assert.Equal(t, weight, link.ConnectionWeight, "Wrong incoming link weight")
	assert.Equal(t, node, link.InNode, "Wrong InNode in Link")
	assert.Equal(t, node2, link.OutNode, "Wrong OutNode in Link")
}

func TestNNode_ConnectFrom(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)

	weight := 1.5
	node2.ConnectFrom(node, weight)
	assert.Len(t, node2.Incoming, 1, "Wrong number of incoming nodes")
	assert.Len(t, node.Outgoing, 1, "wrong number of outgoing links")

	link := node2.Incoming[0]
	assert.Equal(t, weight, link.ConnectionWeight, "Wrong incoming link weight")
	assert.Equal(t, node, link.InNode, "Wrong InNode in Link")
	assert.Equal(t, node2, link.OutNode, "Wrong OutNode in Link")

	assert.EqualValues(t, link, node.Outgoing[0], "incoming and outgoing links do not match")
}

// Tests NNode Depth
func TestNNode_Depth(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)
	node3 := NewNNode(3, OutputNeuron)

	node2.AddIncoming(node, 15.0)
	node3.AddIncoming(node2, 20.0)

	depth, err := node3.Depth(0, 0)
	require.NoError(t, err)
	assert.Equal(t, 2, depth)
}

func TestNNode_DepthWithLoop(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)
	node3 := NewNNode(3, OutputNeuron)

	node2.AddIncoming(node, 15.0)
	node3.AddIncoming(node2, 20.0)
	node3.AddIncoming(node3, 10.0)
	depth, err := node3.Depth(0, 0)
	require.NoError(t, err)
	assert.Equal(t, 2, depth)
}

func TestNNode_Depth_withMaximumLimit(t *testing.T) {
	node := NewNNode(1, InputNeuron)
	node2 := NewNNode(2, HiddenNeuron)
	node3 := NewNNode(3, OutputNeuron)

	node2.AddIncoming(node, 15.0)
	node3.AddIncoming(node2, 20.0)

	maxDepth := 1
	depth, err := node3.Depth(0, maxDepth)
	require.EqualError(t, err, ErrMaximalNetDepthExceeded.Error())
	assert.Equal(t, maxDepth, depth)
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

func TestNNode_GetActiveOut(t *testing.T) {
	activation := 1293.98
	node := NewNNode(1, InputNeuron)
	node.Activation = activation

	out := node.GetActiveOut()
	assert.Equal(t, 0.0, out, "no activations - must be zero")

	node.ActivationsCount = 1
	out = node.GetActiveOut()
	assert.Equal(t, activation, out)
}

func TestNNode_GetActiveOutTd(t *testing.T) {
	activation := 1293.98
	node := NewNNode(1, InputNeuron)
	node.lastActivation = activation
	node.ActivationsCount = 1

	out := node.GetActiveOutTd()
	assert.Equal(t, 0.0, out, "last activation not available yet")

	node.ActivationsCount = 2
	out = node.GetActiveOutTd()
	assert.Equal(t, activation, out)
}

func TestNNode_IsSensor(t *testing.T) {
	testCases := []struct {
		nType    NodeNeuronType
		isSensor bool
	}{
		{
			nType:    InputNeuron,
			isSensor: true,
		},
		{
			nType:    BiasNeuron,
			isSensor: true,
		},
		{
			nType:    HiddenNeuron,
			isSensor: false,
		},
		{
			nType:    OutputNeuron,
			isSensor: false,
		},
	}
	for i, tc := range testCases {
		node := NewNNode(1, tc.nType)
		assert.Equal(t, tc.isSensor, node.IsSensor(), "wrong value at: %d", i)
	}
}

func TestNNode_IsNeuron(t *testing.T) {
	testCases := []struct {
		nType    NodeNeuronType
		isNeuron bool
	}{
		{
			nType:    InputNeuron,
			isNeuron: false,
		},
		{
			nType:    BiasNeuron,
			isNeuron: false,
		},
		{
			nType:    HiddenNeuron,
			isNeuron: true,
		},
		{
			nType:    OutputNeuron,
			isNeuron: true,
		},
	}
	for i, tc := range testCases {
		node := NewNNode(1, tc.nType)
		assert.Equal(t, tc.isNeuron, node.IsNeuron(), "wrong value at: %d", i)
	}
}

func TestNNode_FlushbackCheck(t *testing.T) {
	testCases := []struct {
		node   NNode
		failed bool
	}{
		{
			node:   NNode{},
			failed: false,
		},
		{
			node:   NNode{Activation: 1.0},
			failed: true,
		},
		{
			node:   NNode{ActivationsCount: 1},
			failed: true,
		},
		{
			node:   NNode{lastActivation: 2.0},
			failed: true,
		},
		{
			node:   NNode{lastActivation2: 3.4},
			failed: true,
		},
	}
	for i, tc := range testCases {
		err := tc.node.FlushbackCheck()
		if tc.failed {
			assert.Error(t, err, "error expected at: %d", i)
		} else {
			assert.NoError(t, err, "no error expected at: %d", i)
		}
	}
}

func TestNNode_NodeType(t *testing.T) {
	testCases := []struct {
		neuronType NodeNeuronType
		nodeType   NodeType
	}{
		{
			neuronType: BiasNeuron,
			nodeType:   SensorNode,
		},
		{
			neuronType: InputNeuron,
			nodeType:   SensorNode,
		},
		{
			neuronType: HiddenNeuron,
			nodeType:   NeuronNode,
		},
		{
			neuronType: OutputNeuron,
			nodeType:   NeuronNode,
		},
	}
	for i, tc := range testCases {
		node := NewNNode(1, tc.neuronType)
		assert.Equal(t, tc.nodeType, node.NodeType(), "wrong node type at: %d", i)
	}
}

func TestNNode_PrintDebug(t *testing.T) {
	node := NewNNode(1, InputNeuron)

	out := node.PrintDebug()
	assert.NotEmpty(t, out)
}

func TestNNode_String(t *testing.T) {
	node := NewNNode(1, InputNeuron)

	out := node.String()
	assert.NotEmpty(t, out)
}
