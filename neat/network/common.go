// Package network provides data structures and utilities to describe Artificial Neural Network and network solvers.
package network

import (
	"errors"
	"fmt"
	neatmath "github.com/yaricom/goNEAT/v2/neat/math"
	"math"
)

var (
	// ErrNetExceededMaxActivationAttempts The error to be raised when maximal number of network activation attempts exceeded
	ErrNetExceededMaxActivationAttempts = errors.New("maximal network activation attempts exceeded")
	// ErrNetUnsupportedSensorsArraySize The error to be raised when unsupported size of the sensor data array provided
	ErrNetUnsupportedSensorsArraySize = errors.New("the sensors array size is unsupported by network solver")
	// ErrMaximalNetDepthExceeded The error to be raised when depth of the network exceeds maximal allowed
	ErrMaximalNetDepthExceeded = errors.New("depth of the network exceeds maximum allowed, fallback to maximal")
	// ErrZeroActivationStepsRequested the error to be raised when zero activation steps requested
	ErrZeroActivationStepsRequested = errors.New("zero activation steps requested")
)

// NodeType NNodeType defines the type of NNode to create
type NodeType byte

// Predefined NNode types
const (
	// NeuronNode The neuron type
	NeuronNode NodeType = iota
	// SensorNode The sensor type
	SensorNode
)

// NodeTypeName Returns human-readable NNode type name for given constant value
func NodeTypeName(nType NodeType) string {
	switch nType {
	case NeuronNode:
		return "NEURON"
	case SensorNode:
		return "SENSOR"
	default:
		return "UNKNOWN NODE TYPE"
	}
}

// NodeNeuronType NeuronType defines the type of neuron to create
type NodeNeuronType byte

// These are NNode layer type
const (
	// HiddenNeuron The node is in hidden layer
	HiddenNeuron NodeNeuronType = iota
	// InputNeuron The node is in input layer
	InputNeuron
	// OutputNeuron The node is in output layer
	OutputNeuron
	// BiasNeuron The node is bias
	BiasNeuron
)

const (
	hiddenNeuronName = "HIDN"
	inputNeuronName  = "INPT"
	outputNeuronName = "OUTP"
	biasNeuronName   = "BIAS"
	unknownNeuroName = "UNKNOWN NEURON TYPE"
)

// NeuronTypeName Returns human-readable neuron type name for given constant
func NeuronTypeName(neuronType NodeNeuronType) string {
	switch neuronType {
	case HiddenNeuron:
		return hiddenNeuronName
	case InputNeuron:
		return inputNeuronName
	case OutputNeuron:
		return outputNeuronName
	case BiasNeuron:
		return biasNeuronName
	default:
		return unknownNeuroName
	}
}

// NeuronTypeByName Returns neuron node type from its name
func NeuronTypeByName(name string) (NodeNeuronType, error) {
	switch name {
	case hiddenNeuronName:
		return HiddenNeuron, nil
	case inputNeuronName:
		return InputNeuron, nil
	case outputNeuronName:
		return OutputNeuron, nil
	case biasNeuronName:
		return BiasNeuron, nil
	default:
		return math.MaxInt8, errors.New("Unknown neuron type name: " + name)
	}
}

// ActivateNode Method to calculate activation for specified neuron node based on it's ActivationType field value.
// Will return error and set -0.0 activation if unsupported activation type requested.
func ActivateNode(node *NNode, a *neatmath.NodeActivatorsFactory) error {
	out, err := a.ActivateByType(node.ActivationSum, node.Params, node.ActivationType)
	if err == nil {
		node.setActivation(out)
	}
	return err
}

// ActivateModule Method to activate neuron module presented by provided node. As a result of execution the activation values of all
// input nodes will be processed by corresponding activation function and corresponding activation values of output nodes
// will be set. Will panic if unsupported activation type requested.
func ActivateModule(module *NNode, a *neatmath.NodeActivatorsFactory) error {
	inputs := make([]float64, len(module.Incoming))
	for i, v := range module.Incoming {
		inputs[i] = v.InNode.GetActiveOut()
	}

	outputs, err := a.ActivateModuleByType(inputs, module.Params, module.ActivationType)
	if err != nil {
		return err
	}
	if len(outputs) != len(module.Outgoing) {
		return fmt.Errorf(
			"number of output parameters [%d] returned by module activator doesn't match "+
				"the number of output neurons of the module [%d]", len(outputs), len(module.Outgoing))
	}
	// set outputs
	for i, out := range outputs {
		module.Outgoing[i].OutNode.setActivation(out)
		module.Outgoing[i].OutNode.isActive = true // activate output node
	}
	return nil
}

// NodeIdGenerator definition of the unique IDs generator for network nodes.
type NodeIdGenerator interface {
	// NextNodeId is to get next unique node ID
	NextNodeId() int
}
