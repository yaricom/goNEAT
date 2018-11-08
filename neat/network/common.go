// The package network provides data holders and utilities to describe Artificial Neural Network
package network

import (
	"math"
	"fmt"
)

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

// NodeActivationType defines the type of activation function to use for the neuron node
type NodeActivationType byte

// The neuron Activation function Types
const (
	// The sigmoid activation functions
	SigmoidPlainActivation NodeActivationType = iota + 1
	SigmoidReducedActivation
	SigmoidBipolarActivation
	SigmoidSteepenedActivation
	SigmoidApproximationActivation
	SigmoidSteepenedApproximationActivation
	SigmoidInverseAbsoluteActivation
	SigmoidLeftShiftedActivation
	SigmoidLeftShiftedSteepenedActivation
	SigmoidRightShiftedSteepenedActivation

	// The other activators assortment
	TanhActivation
	GaussianBipolarActivation
	LinearAbsActivation
	LinearClippedActivation
	NullFnActivation
	SignActivation
	SineActivation
	StepActivation
)

// The activation function type
type ActivationFunction func(float64, []float64) float64

// The default node activators factory reference
var NodeActivators = NewNodeActivatorsFactory()

// The factory to provide appropriate neuron node activation function
type NodeActivatorsFactory struct {
	// The map of registered activators by type
	activators map[NodeActivationType]ActivationFunction

	// The forward and inverse maps of activator type and function name
	forward    map[NodeActivationType]string
	inverse    map[string]NodeActivationType
}

// Returns node activator factory initialized with default activation functions
func NewNodeActivatorsFactory() *NodeActivatorsFactory {
	af := &NodeActivatorsFactory{
		activators:make(map[NodeActivationType]ActivationFunction),
		forward:make(map[NodeActivationType]string),
		inverse:make(map[string]NodeActivationType),
	}
	af.Register(SigmoidPlainActivation, plainSigmoid, "SigmoidPlainActivation")
	af.Register(SigmoidReducedActivation, reducedSigmoid, "SigmoidReducedActivation")
	af.Register(SigmoidSteepenedActivation, steepenedSigmoid, "SigmoidSteepenedActivation")
	af.Register(SigmoidBipolarActivation, bipolarSigmoid, "SigmoidBipolarActivation")
	af.Register(SigmoidApproximationActivation, approximationSigmoid, "SigmoidApproximationActivation")
	af.Register(SigmoidSteepenedApproximationActivation, approximationSteepenedSigmoid, "SigmoidSteepenedApproximationActivation")
	af.Register(SigmoidInverseAbsoluteActivation, inverseAbsoluteSigmoid, "SigmoidInverseAbsoluteActivation")
	af.Register(SigmoidLeftShiftedActivation, leftShiftedSigmoid, "SigmoidLeftShiftedActivation")
	af.Register(SigmoidLeftShiftedSteepenedActivation, leftShiftedSteepenedSigmoid, "SigmoidLeftShiftedSteepenedActivation")
	af.Register(SigmoidRightShiftedSteepenedActivation, rightShiftedSteepenedSigmoid, "SigmoidRightShiftedSteepenedActivation")

	af.Register(TanhActivation, hyperbolicTangent, "TanhActivation")
	af.Register(GaussianBipolarActivation, bipolarGaussian, "GaussianBipolarActivation")
	af.Register(LinearAbsActivation, absoluteLinear, "LinearAbsActivation")
	af.Register(LinearClippedActivation, clippedLinear, "LinearClippedActivation")
	af.Register(NullFnActivation, nullFunctor, "NullFnActivation")
	af.Register(SignActivation, signFunctor, "SignActivation")
	af.Register(SineActivation, sineFunctor, "SineActivation")
	af.Register(StepActivation, stepFunction, "StepActivation")

	return af
}

// Method to calculate activation for specified neuron node based on it's ActivationType field value.
// Will panic if unsupported activation type requested.
func (a *NodeActivatorsFactory) ActivateNode(node *NNode) float64 {
	if fn, ok := a.activators[node.ActivationType]; ok {
		return fn(node.ActivationSum, node.Params)
	} else {
		panic("Unknown activation type")
	}
}

// Method to calculate activation value for give input and auxiliary parameters using activation function with specified type.
// Will panic if unsupported activation type requested.
func (a *NodeActivatorsFactory) ActivateByType(input float64, aux_params[]float64, a_type NodeActivationType) float64 {
	if fn, ok := a.activators[a_type]; ok {
		return fn(input, aux_params)
	} else {
		panic("Unknown activation type")
	}
}

// Registers given function with provided type and name into the factory
func (a *NodeActivatorsFactory) Register(a_type NodeActivationType, a_func ActivationFunction, f_name string) {
	// store function
	a.activators[a_type] = a_func
	// store name<->type bi-directional mapping
	a.forward[a_type] = f_name
	a.inverse[f_name] = a_type
}

// Parse node activation type name and return corresponding activation type
func (a *NodeActivatorsFactory) ActivationTypeFromName(name string) NodeActivationType {
	if t, ok := a.inverse[name]; ok {
		return t
	} else {
		panic("Unsupported activation type name: " + name)
	}
}

// Returns activation function name from given type
func (a *NodeActivatorsFactory) ActivationNameFromType(atype NodeActivationType) string {
	if n, ok := a.forward[atype]; ok {
		return n
	} else {
		panic(fmt.Sprintf("Unsupported activation type: %d", atype))
	}
}

// The sigmoid activation functions
var (
	// The plain sigmoid
	plainSigmoid = func(input float64, aux_params[]float64) float64 {
		return (1 / (1 + math.Exp(-input)))
	}
	// The reduced sigmoid
	reducedSigmoid = func(input float64, aux_params[]float64) float64 {
		return (1 / (1 + math.Exp(-0.5 * input)))
	}
	// The steepened sigmoid
	steepenedSigmoid = func(input float64, aux_params[]float64) float64 {
		return 1.0 / (1.0 + math.Exp(-4.924273 * input))
	}
	// The bipolar sigmoid activation function xrange->[-1,1] yrange->[-1,1]
	bipolarSigmoid = func(input float64, aux_params[]float64) float64 {
		return (2.0 / (1.0 + math.Exp(-4.924273 * input))) - 1.0
	}
	// The approximation sigmoid with squashing range [-4.0; 4.0]
	approximationSigmoid = func(input float64, aux_params[]float64) float64 {
		four, one_32nd := float64(4.0), float64(0.03125)
		if input < -4.0 {
			return 0.0
		} else if input < 0.0 {
			return (input + four) * (input + four) * one_32nd
		} else if input < 4.0 {
			return 1.0 - (input - four) * (input - four) * one_32nd
		} else {
			return 1.0
		}
	}
	// The steepened aproximation sigmoid with squashing range [-1.0; 1.0]
	approximationSteepenedSigmoid = func(input float64, aux_params[]float64) float64 {
		one, one_half := 1.0, 0.5
		if input < -1.0 {
			return 0.0
		} else if input < 0.0 {
			return (input + one) * (input + one) * one_half
		} else if input < 1.0 {
			return 1.0 - (input - one) * (input - one) * one_half
		} else {
			return 1.0;
		}
	}
	// The inverse absolute sigmoid
	inverseAbsoluteSigmoid = func(input float64, aux_params[]float64) float64 {
		return 0.5 + (input / (1.0 + math.Abs(input))) * 0.5
	}

	// The left/right shifted sigmoids
	leftShiftedSigmoid = func(input float64, aux_params[]float64) float64 {
		return 1.0 / (1.0 + math.Exp(-input - 2.4621365))
	}
	leftShiftedSteepenedSigmoid = func(input float64, aux_params[]float64) float64 {
		return 1.0 / (1.0 + math.Exp(-(4.924273 * input + 2.4621365)))
	}
	rightShiftedSteepenedSigmoid = func(input float64, aux_params[]float64) float64 {
		return 1.0 / (1.0 + math.Exp(-(4.924273 * input - 2.4621365)))
	}
)

// The other activation functions
var (
	// The hyperbolic tangent
	hyperbolicTangent = func(input float64, aux_params[]float64) float64 {
		return math.Tanh(0.9 * input)
	}
	// The bipolar Gaussian activator xrange->[-1,1] yrange->[-1,1]
	bipolarGaussian = func(input float64, aux_params[]float64) float64 {
		return 2.0 * math.Exp(-math.Pow(input * 2.5, 2.0)) - 1.0
	}
	// The absolute linear
	absoluteLinear = func(input float64, aux_params[]float64) float64 {
		return math.Abs(input)
	}
	// Linear activation function with clipping. By 'clipping' we mean the output value is linear between
	/// x = -1 and x = 1. Below -1 and above +1 the output is clipped at -1 and +1 respectively
	clippedLinear = func(input float64, aux_params[]float64) float64 {
		if (input < -1.0) {
			return -1.0
		}
		if (input > 1.0) {
			return 1.0
		}
		return input
	}
	// The null activator
	nullFunctor = func(input float64, aux_params[]float64) float64 {
		return 0.0
	}
	// The sign activator
	signFunctor = func(input float64, aux_params[]float64) float64 {
		if math.IsNaN(input) || input == 0.0 {
			return 0.0
		} else if math.Signbit(input) {
			return -1.0
		} else {
			return 1.0
		}
	}
	// The sine periodic activation with doubled period
	sineFunctor = func(input float64, aux_params[]float64) float64 {
		return math.Sin(2.0 * input)
	}
	// The step function x<0 ? 0.0 : 1.0
	stepFunction = func(input float64, aux_params[]float64) float64 {
		if math.Signbit(input) {
			return 0.0
		} else {
			return 1.0
		}
	}
)
