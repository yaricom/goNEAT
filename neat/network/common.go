// The package network provides data holders and utilities to describe Artificial Neural Network
package network

import "math"

// NNodeType defines the type of NNode to create
type NodeType byte

// Predefined NNode types
const (
	// The neuron type
	NeuronNode NodeType = iota
	// The sensor type
	SensorNode
)

// Returns human readable NNode type name for given constant value
func NodeTypeName(ntype NodeType) string {
	switch ntype {
	case NeuronNode:
		return "NEURON"
	case SensorNode:
		return "SENSOR"
	default:
		return "!!! UNKNOWN NODE TYPE !!!"
	}
}

// NeuronType defines the type of neuron to create
type NeuronType byte

// These are NNode layer type
const (
	// The node is in hidden layer
	HiddenNeuron NeuronType = iota
	// The node is in input layer
	InputNeuron
	// The node is in output layer
	OutputNeuron
	// The node is bias
	BiasNeuron
)

// Returns human readable neuron type name for given constant
func NeuronTypeName(nlayer NeuronType) string {
	switch nlayer {
	case HiddenNeuron:
		return "HIDDEN"
	case InputNeuron:
		return "INPUT"
	case OutputNeuron:
		return "OUTPUT"
	case BiasNeuron:
		return "BIAS"
	default:
		return "!!! UNKNOWN NEURON TYPE !!!"
	}
}

// ActivationType defines the type of activation function to use for the neuron
type ActivationType byte

// The neuron Activation function Types
const (
	// The non-shifted stepened sigmoid activation function
	SigmoidSteepened ActivationType = iota + 1
	SigmoidLeftShifted
	SigmoidLeftShiftedSteepened
	SigmoidRightShiftedSteepened
	Sigmoid
	Tanh
	InverseAbs
)

// The collection of activation functions
func activate(node *NNode) float64 {
	switch node.ActivationType {
	case SigmoidSteepened:
		return 1.0 / (1.0 + math.Exp(-4.924273 * node.ActivationSum)) //Compressed
	case SigmoidLeftShifted:
		return (1 / (1 + math.Exp(-node.ActivationSum - 2.4621365)))
	case SigmoidLeftShiftedSteepened:
		return (1 / (1 + math.Exp(-(4.924273 * node.ActivationSum + 2.4621365))))
	case SigmoidRightShiftedSteepened:
		return (1 / (1 + math.Exp(-(4.924273 * node.ActivationSum - 2.4621365))))
	case Sigmoid:
		return (1 / (1 + math.Exp(-node.ActivationSum)))
	case Tanh:
		return math.Tanh(0.9 * node.ActivationSum)
	case InverseAbs:
		return node.ActivationSum / (1.0 + math.Abs(node.ActivationSum))
	default:
		panic("Unknown activation type")
	}
}