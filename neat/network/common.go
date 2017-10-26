// The package network provides data holders and utilities to describe Artificial Neural Network
package network

import "math"

// NNodeType defines the type of NNode to create
type NodeType byte

// Predefined NNode types
const (
	// The neuron type
	NEURON NodeType = iota
	// The sensor type
	SENSOR
)

// Returns human readable NNode type name for given constant value
func NodeTypeName(ntype NodeType) string {
	switch ntype {
	case NEURON:
		return "NEURON"
	case SENSOR:
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
	HIDDEN NeuronType = iota
	// The node is in input layer
	INPUT
	// The node is in output layer
	OUTPUT
	// The node is bias
	BIAS
)

// Returns human readable neuron type name for given constant
func NeuronTypeName(nlayer NeuronType) string {
	switch nlayer {
	case HIDDEN:
		return "HIDDEN"
	case INPUT:
		return "INPUT"
	case OUTPUT:
		return "OUTPUT"
	case BIAS:
		return "BIAS"
	default:
		return "!!! UNKNOWN NEURON TYPE !!!"
	}
}

// ActivationType defines the type of activation function to use for the neuron
type ActivationType byte

// The neuron Activation function Types
const (
	// The sigmoid activation function
	SIGMOID ActivationType = iota
)

// The neuron activator function
type ActivationFunc func(node *NNode, slope, constant float64) float64

// handling neuron activation calculation
func (f ActivationFunc) Activation(node *NNode, slope, constant float64) float64 {
	return f(node, slope, constant)
}

// SIGMOID FUNCTION ********************************
// This is a signmoidal activation function, which is an S-shaped squashing function.
// It smoothly limits the amplitude of the output of a neuron to between 0 and 1.
// It is a helper to the neural-activation function get_active_out.
// It is made inline so it can execute quickly since it is at every non-sensor node in a network.
// NOTE:  In order to make node insertion in the middle of a link possible,
// the signmoid can be shifted to the right and more steeply sloped:
// slope=4.924273
// constant= 2.4621365
// These parameters optimize mean squared error between the old output,
// and an output of a node inserted in the middle of a link between
// the old output and some other node.
// When not right-shifted, the steepened slope is closest to a linear
// ascent as possible between -0.5 and 0.5
func SigmoidActivation(node *NNode, slope, constant float64) float64 {
	//RIGHT SHIFTED ---------------------------------------------------------
	//return (1/(1+(exp(-(slope*activesum-constant))))); //ave 3213 clean on 40 runs of p2m and 3468 on another 40
	//41394 with 1 failure on 8 runs

	//LEFT SHIFTED ----------------------------------------------------------
	//return (1/(1+(exp(-(slope*activesum+constant))))); //original setting ave 3423 on 40 runs of p2m, 3729 and 1 failure also

	//PLAIN SIGMOID ---------------------------------------------------------
	//return (1/(1+(exp(-activesum)))); //3511 and 1 failure

	//LEFT SHIFTED NON-STEEPENED---------------------------------------------
	//return (1/(1+(exp(-activesum-constant)))); //simple left shifted

	//NON-SHIFTED STEEPENED
	return 1.0 / (1.0 + (math.Exp(-(slope * node.ActivationSum)))) //Compressed
}