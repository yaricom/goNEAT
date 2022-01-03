package network

import (
	"errors"
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"testing"
)

const alwaysErrorText = "always be failing"

var alwaysError = errors.New(alwaysErrorText)

type ErrorWriter int

func (e ErrorWriter) Write(_ []byte) (int, error) {
	return 0, alwaysError
}

func TestNodeTypeName(t *testing.T) {
	name := NodeTypeName(NeuronNode)
	assert.Equal(t, "NEURON", name)
	name = NodeTypeName(SensorNode)
	assert.Equal(t, "SENSOR", name)
	name = NodeTypeName(SensorNode + 1)
	assert.Equal(t, "UNKNOWN NODE TYPE", name)
}

func TestNeuronTypeName(t *testing.T) {
	name := NeuronTypeName(HiddenNeuron)
	assert.Equal(t, hiddenNeuronName, name)
	name = NeuronTypeName(InputNeuron)
	assert.Equal(t, inputNeuronName, name)
	name = NeuronTypeName(OutputNeuron)
	assert.Equal(t, outputNeuronName, name)
	name = NeuronTypeName(BiasNeuron)
	assert.Equal(t, biasNeuronName, name)
	name = NeuronTypeName(BiasNeuron + 1)
	assert.Equal(t, unknownNeuroName, name)
}

func TestNeuronTypeByName(t *testing.T) {
	nType, err := NeuronTypeByName(hiddenNeuronName)
	assert.NoError(t, err)
	assert.Equal(t, HiddenNeuron, nType)
	nType, err = NeuronTypeByName(inputNeuronName)
	assert.NoError(t, err)
	assert.Equal(t, InputNeuron, nType)
	nType, err = NeuronTypeByName(outputNeuronName)
	assert.NoError(t, err)
	assert.Equal(t, OutputNeuron, nType)
	nType, err = NeuronTypeByName(biasNeuronName)
	assert.NoError(t, err)
	assert.Equal(t, BiasNeuron, nType)
	nType, err = NeuronTypeByName(unknownNeuroName)
	assert.EqualError(t, err, "Unknown neuron type name: "+unknownNeuroName)
	assert.Equal(t, NodeNeuronType(1<<7-1), nType)
}

func TestActivateNode(t *testing.T) {
	node := NewNNode(1, BiasNeuron)
	err := ActivateNode(node, math.NodeActivators)
	assert.NoError(t, err)

	node.ActivationType = math.MinModuleActivation + 1
	err = ActivateNode(node, math.NodeActivators)
	assert.EqualError(t, err, fmt.Sprintf("unknown neuron activation type: %d", node.ActivationType))
}

func TestActivateModule(t *testing.T) {
	node := NewNNode(1, HiddenNeuron)
	node.ActivationType = math.MultiplyModuleActivation
	node.AddIncoming(NewNNode(2, HiddenNeuron), 1.0)
	node.AddIncoming(NewNNode(3, HiddenNeuron), 1.0)
	node.AddOutgoing(NewNNode(4, HiddenNeuron), 1.0)

	err := ActivateModule(node, math.NodeActivators)
	assert.NoError(t, err)

	node.ActivationType = math.MinModuleActivation + 1
	err = ActivateModule(node, math.NodeActivators)
	assert.EqualError(t, err, fmt.Sprintf("unknown module activation type: %d", node.ActivationType))

	// add extra outgoing
	node.AddOutgoing(NewNNode(5, HiddenNeuron), 1.0)
	node.ActivationType = math.MultiplyModuleActivation
	err = ActivateModule(node, math.NodeActivators)
	assert.EqualError(t, err, fmt.Sprintf(
		"number of output parameters [%d] returned by module activator doesn't match "+
			"the number of output neurons of the module [%d]", 1, len(node.Outgoing)))
}
