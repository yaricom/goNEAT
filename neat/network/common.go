// The package network provides data holders and utilities to describe Artificial Neural Network
package network

import (
	"math"
	"fmt"
	"errors"
	"github.com/yaricom/goNEAT/neat/utils"
)

var (
	// The error to be raised when maximal number of network activation attempts exceeded
	NetErrExceededMaxActivationAttempts = errors.New("maximal network activation attempts exceeded.")
	// The error to be raised when unsupported sensors data array size provided
	NetErrUnsupportedSensorsArraySize = errors.New("the sensors array size is unsupported by network solver")
	// The error to be raised when depth calculation failed due to the loop in network
	NetErrDepthCalculationFailedLoopDetected = errors.New("depth can not be determined for network with loop")
)

// Defines network solver interface which describes neural network structures with methods to run activation waves through
// them.
type NetworkSolver interface {
	// Propagates activation wave through all network nodes provided number of steps in forward direction.
	// Returns true if activation wave passed from all inputs to outputs.
	ForwardSteps(steps int) (bool, error)

	// Propagates activation wave through all network nodes provided number of steps by recursion from output nodes
	// Returns true if activation wave passed from all inputs to outputs.
	RecursiveSteps() (bool, error)

	// Attempts to relax network given amount of steps until giving up. The network considered relaxed when absolute
	// value of the change at any given point is less than maxAllowedSignalDelta during activation waves propagation.
	// If maxAllowedSignalDelta value is less than or equal to 0, the method will return true without checking for relaxation.
	Relax(maxSteps int, maxAllowedSignalDelta float64) (bool, error)

	// Flushes network state by removing all current activations. Returns true if network flushed successfully or
	// false in case of error.
	Flush() (bool, error)

	// Set sensors values to the input nodes of the network
	LoadSensors(inputs []float64) error
	// Read output values from the output nodes of the network
	ReadOutputs() []float64

	// Returns the total number of neural units in the network
	NodeCount() int
	// Returns the total number of links between nodes in the network
	LinkCount() int
}

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
type NodeNeuronType byte

// These are NNode layer type
const (
	// The node is in hidden layer
	HiddenNeuron NodeNeuronType = iota
	// The node is in input layer
	InputNeuron
	// The node is in output layer
	OutputNeuron
	// The node is bias
	BiasNeuron
)

// Returns human readable neuron type name for given constant
func NeuronTypeName(nlayer NodeNeuronType) string {
	switch nlayer {
	case HiddenNeuron:
		return "HIDN"
	case InputNeuron:
		return "INPT"
	case OutputNeuron:
		return "OUTP"
	case BiasNeuron:
		return "BIAS"
	default:
		return "!!! UNKNOWN NEURON TYPE !!!"
	}
}

// Returns neuron node type from its name
func NeuronTypeByName(name string) (NodeNeuronType, error) {
	switch name {
	case "HIDN":
		return HiddenNeuron, nil
	case "INPT":
		return InputNeuron, nil
	case "OUTP":
		return OutputNeuron, nil
	case "BIAS":
		return BiasNeuron, nil
	default:
		return math.MaxInt8, errors.New("Unknown neuron type name: " + name)
	}
}

// Method to calculate activation for specified neuron node based on it's ActivationType field value.
// Will return error and set -0.0 activation if unsupported activation type requested.
func ActivateNode(node *NNode, a *utils.NodeActivatorsFactory) (err error) {
	out, err := a.ActivateByType(node.ActivationSum, node.Params, node.ActivationType)
	if err == nil {
		node.setActivation(out)
	}
	return err
}


// Method to activate neuron module presented by provided node. As a result of execution the activation values of all
// input nodes will be processed by corresponding activation function and corresponding activation values of output nodes
// will be set. Will panic if unsupported activation type requested.
func ActivateModule(module *NNode, a *utils.NodeActivatorsFactory) error {
	inputs := make([]float64, len(module.Incoming))
	for i, v := range module.Incoming {
		inputs[i] = v.InNode.GetActiveOut()
	}

	outputs, err := a.ActivateModuleByType(inputs, module.Params, module.ActivationType)
	if err != nil {
		return err
	}
	if len(outputs) != len(module.Outgoing) {
		return errors.New(fmt.Sprintf(
			"The number of output parameters [%d] returned by module activator doesn't match " +
				"the number of output neurons of the module [%d]", len(outputs), len(module.Outgoing)))
	}
	// set outputs
	for i, out := range outputs {
		module.Outgoing[i].OutNode.setActivation(out)
		module.Outgoing[i].OutNode.isActive = true // activate output node
	}
	return nil
}

