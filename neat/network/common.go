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
	GaussianActivation
	LinearActivation
	NullFnActivation
	SignActivation
	SineActivation
	StepActivation
)

// The default node activators factory reference
var defaultNodeActivators = NewNodeActivatorsFactory()

// The factory to provide appropriate neuron node activation function
type NodeActivatorsFactory struct {
	// The map of registered activators by type
	Activators map[NodeActivationType]func(node *NNode) float64
}

// Returns node activator factory initialized with default activation functions
func NewNodeActivatorsFactory() NodeActivatorsFactory {
	af := NodeActivatorsFactory{
		Activators:make(map[NodeActivationType]func(node *NNode) float64),
	}
	af.Activators[SigmoidPlainActivation] = plainSigmoid
	af.Activators[SigmoidReducedActivation] = reducedSigmoid
	af.Activators[SigmoidSteepenedActivation] = steepenedSigmoid
	af.Activators[SigmoidBipolarActivation] = bipolarSigmoid
	af.Activators[SigmoidApproximationActivation] = approximationSigmoid
	af.Activators[SigmoidSteepenedApproximationActivation] = approximationSteepenedSigmoid
	af.Activators[SigmoidInverseAbsoluteActivation] = inverseAbsoluteSigmoid
	af.Activators[SigmoidLeftShiftedActivation] = leftShiftedSigmoid
	af.Activators[SigmoidLeftShiftedSteepenedActivation] = leftShiftedSteepenedSigmoid
	af.Activators[SigmoidRightShiftedSteepenedActivation] = rightShiftedSteepenedSigmoid

	af.Activators[TanhActivation] = hyperbolicTangent
	af.Activators[GaussianActivation] = symmetricGaussian
	af.Activators[LinearActivation] = absoluteLinear
	af.Activators[NullFnActivation] = nullFunctor
	af.Activators[SignActivation] = signFunctor
	af.Activators[SineActivation] = sineFunctor
	af.Activators[StepActivation] = stepFunction

	return af
}

// Method to calculate activation for specified neuron node based on it's ActivationType field value.
// Will panic if unsupported activation type requested.
func (a *NodeActivatorsFactory) Activate(node *NNode) float64 {
	if fn, ok := a.Activators[node.ActivationType]; ok {
		return fn(node)
	} else {
		panic("Unknown activation type")
	}
}

// Parse node activation type name and return corresponding activation type
func ActivationTypeFromName(name string) NodeActivationType {
	switch name {
	case "SigmoidPlainActivation":
		return SigmoidPlainActivation
	case "SigmoidReducedActivation":
		return SigmoidReducedActivation
	case "SigmoidBipolarActivation":
		return SigmoidBipolarActivation
	case "SigmoidSteepenedActivation":
		return SigmoidSteepenedActivation
	case "SigmoidApproximationActivation":
		return SigmoidApproximationActivation
	case "SigmoidSteepenedApproximationActivation":
		return SigmoidSteepenedApproximationActivation
	case "SigmoidInverseAbsoluteActivation":
		return SigmoidInverseAbsoluteActivation
	case "SigmoidLeftShiftedActivation":
		return SigmoidLeftShiftedActivation
	case "SigmoidLeftShiftedSteepenedActivation":
		return SigmoidLeftShiftedSteepenedActivation
	case "SigmoidRightShiftedSteepenedActivation":
		return SigmoidRightShiftedSteepenedActivation
	case "TanhActivation":
		return TanhActivation
	case "GaussianActivation":
		return GaussianActivation
	case "LinearActivation":
		return LinearActivation
	case "NullFnActivation":
		return NullFnActivation
	case "SignActivation":
		return SignActivation
	case "SineActivation":
		return SineActivation
	case "StepActivation":
		return StepActivation
	default:
		panic("Unsupported activation type name: " + name)
	}
}

// Returns activation function name from given type
func ActivationNameFromType(atype NodeActivationType) string {
	switch atype {
	case SigmoidPlainActivation:
		return "SigmoidPlainActivation"
	case SigmoidReducedActivation:
		return "SigmoidReducedActivation"
	case SigmoidBipolarActivation:
		return "SigmoidBipolarActivation"
	case SigmoidSteepenedActivation:
		return "SigmoidSteepenedActivation"
	case SigmoidApproximationActivation:
		return "SigmoidApproximationActivation"
	case SigmoidSteepenedApproximationActivation:
		return "SigmoidSteepenedApproximationActivation"
	case SigmoidInverseAbsoluteActivation:
		return "SigmoidInverseAbsoluteActivation"
	case SigmoidLeftShiftedActivation:
		return "SigmoidLeftShiftedActivation"
	case SigmoidLeftShiftedSteepenedActivation:
		return "SigmoidLeftShiftedSteepenedActivation"
	case SigmoidRightShiftedSteepenedActivation:
		return "SigmoidRightShiftedSteepenedActivation"
	case TanhActivation:
		return "TanhActivation"
	case GaussianActivation:
		return "GaussianActivation"
	case LinearActivation:
		return "LinearActivation"
	case NullFnActivation:
		return "NullFnActivation"
	case SignActivation:
		return "SignActivation"
	case SineActivation:
		return "SineActivation"
	case StepActivation:
		return "StepActivation"
	default:
		panic(fmt.Sprintf("Unsupported activation type: %d", atype))
	}
}

// The sigmoid activation functions
var (
	// The plain sigmoid
	plainSigmoid = func(node *NNode) float64 {
		return (1 / (1 + math.Exp(-node.ActivationSum)))
	}
	// The reduced sigmoid
	reducedSigmoid = func(node *NNode) float64 {
		return (1 / (1 + math.Exp(-0.5 * node.ActivationSum)))
	}
	// The steepened sigmoid
	steepenedSigmoid = func(node *NNode) float64 {
		return 1.0 / (1.0 + math.Exp(-4.924273 * node.ActivationSum))
	}
	// The bipolar sigmoid activation function
	bipolarSigmoid = func(node *NNode) float64 {
		return (2.0 / (1.0 + math.Exp(-4.924273 * node.ActivationSum))) - 1.0
	}
	// The approximation sigmoid with squashing range [-4.0; 4.0]
	approximationSigmoid = func(node *NNode) float64 {
		four, one_32nd := float64(4.0), float64(0.03125)
		if node.ActivationSum < -4.0 {
			return 0.0
		} else if node.ActivationSum < 0.0 {
			return (node.ActivationSum + four) * (node.ActivationSum + four) * one_32nd
		} else if node.ActivationSum < 4.0 {
			return 1.0 - (node.ActivationSum - four) * (node.ActivationSum - four) * one_32nd
		} else {
			return 1.0
		}
	}
	// The steepened aproximation sigmoid with squashing range [-1.0; 1.0]
	approximationSteepenedSigmoid = func(node *NNode) float64 {
		one, one_half := 1.0, 0.5
		if node.ActivationSum < -1.0 {
			return 0.0
		} else if node.ActivationSum < 0.0 {
			return (node.ActivationSum + one) * (node.ActivationSum + one) * one_half
		} else if node.ActivationSum < 1.0 {
			return 1.0 - (node.ActivationSum - one) * (node.ActivationSum - one) * one_half
		} else {
			return 1.0;
		}
	}
	// The inverse absolute sigmoid
	inverseAbsoluteSigmoid = func(node *NNode) float64 {
		return 0.5 + (node.ActivationSum / (1.0 + math.Abs(node.ActivationSum))) * 0.5
	}

	// The left/right shifted sigmoids
	leftShiftedSigmoid = func(node *NNode) float64 {
		return 1.0 / (1.0 + math.Exp(-node.ActivationSum - 2.4621365))
	}
	leftShiftedSteepenedSigmoid = func(node *NNode) float64 {
		return 1.0 / (1.0 + math.Exp(-(4.924273 * node.ActivationSum + 2.4621365)))
	}
	rightShiftedSteepenedSigmoid = func(node *NNode) float64 {
		return 1.0 / (1.0 + math.Exp(-(4.924273 * node.ActivationSum - 2.4621365)))
	}
)

// The other activation functions
var (
	// The hyperbolic tangent
	hyperbolicTangent = func(node *NNode) float64 {
		return math.Tanh(0.9 * node.ActivationSum)
	}
	// The symmetric Gaussian activator
	symmetricGaussian = func(node *NNode) float64 {
		return 2.0 * math.Exp(-math.Pow(node.ActivationSum * 2.5, 2.0)) - 1.0
	}
	// The absolute linear
	absoluteLinear = func(node *NNode) float64 {
		return math.Abs(node.ActivationSum)
	}
	// The null activator
	nullFunctor = func(node *NNode) float64 {
		return 0.0
	}
	// The sign activator
	signFunctor = func(node *NNode) float64 {
		if math.IsNaN(node.ActivationSum) || node.ActivationSum == 0.0 {
			return 0.0
		} else if math.Signbit(node.ActivationSum) {
			return -1.0
		} else {
			return 1.0
		}
	}
	// The sine periodic activation with doubled period
	sineFunctor = func(node *NNode) float64 {
		return math.Sin(2.0 * node.ActivationSum)
	}
	// The step function x<0 ? 0.0 : 1.0
	stepFunction = func(node *NNode) float64 {
		if math.Signbit(node.ActivationSum) {
			return 0.0
		} else {
			return 1.0
		}
	}
)
