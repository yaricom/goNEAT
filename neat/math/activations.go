package math

import (
	"fmt"
	"math"
)

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
	GaussianActivation
	LinearActivation
	LinearAbsActivation
	LinearClippedActivation
	NullActivation
	SignActivation
	SineActivation
	StepActivation

	// The modular activators (with multiple inputs/outputs)
	MultiplyModuleActivation
	MaxModuleActivation
	MinModuleActivation
)

// ActivationFunction The neuron node activation function type
type ActivationFunction func(float64, []float64) float64

// ModuleActivationFunction The neurons module activation function type
type ModuleActivationFunction func([]float64, []float64) []float64

// NodeActivators The default node activators factory reference
var NodeActivators = NewNodeActivatorsFactory()

// NodeActivatorsFactory The factory to provide appropriate neuron node activation function
type NodeActivatorsFactory struct {
	// The map of registered neuron node activators by type
	activators map[NodeActivationType]ActivationFunction
	// The map of registered neuron module activators by type
	moduleActivators map[NodeActivationType]ModuleActivationFunction

	// The forward and inverse maps of activator type and function name
	forward map[NodeActivationType]string
	inverse map[string]NodeActivationType
}

// NewNodeActivatorsFactory Returns node activator factory initialized with default activation functions
func NewNodeActivatorsFactory() *NodeActivatorsFactory {
	af := &NodeActivatorsFactory{
		activators:       make(map[NodeActivationType]ActivationFunction),
		moduleActivators: make(map[NodeActivationType]ModuleActivationFunction),
		forward:          make(map[NodeActivationType]string),
		inverse:          make(map[string]NodeActivationType),
	}
	// Register neuron node activators
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
	af.Register(GaussianActivation, gaussian, "GaussianActivation")
	af.Register(LinearActivation, linear, "LinearActivation")
	af.Register(LinearAbsActivation, absoluteLinear, "LinearAbsActivation")
	af.Register(LinearClippedActivation, clippedLinear, "LinearClippedActivation")
	af.Register(NullActivation, nullFunctor, "NullActivation")
	af.Register(SignActivation, signFunction, "SignActivation")
	af.Register(SineActivation, sineFunction, "SineActivation")
	af.Register(StepActivation, stepFunction, "StepActivation")

	// register neuron modules activators
	af.RegisterModule(MultiplyModuleActivation, multiplyModule, "MultiplyModuleActivation")
	af.RegisterModule(MaxModuleActivation, maxModule, "MaxModuleActivation")
	af.RegisterModule(MinModuleActivation, minModule, "MinModuleActivation")

	return af
}

// ActivateByType is to calculate activation value for give input and auxiliary parameters using activation function with specified type.
// Will return error and -math.Inf activation if unsupported activation type requested.
func (a *NodeActivatorsFactory) ActivateByType(input float64, auxParams []float64, aType NodeActivationType) (float64, error) {
	if fn, ok := a.activators[aType]; ok {
		return fn(input, auxParams), nil
	} else {
		return math.Inf(-1), fmt.Errorf("unknown neuron activation type: %d", aType)
	}
}

// ActivateModuleByType will apply corresponding module activation function to the input values and returns appropriate output values.
// Will panic if unsupported activation function requested
func (a *NodeActivatorsFactory) ActivateModuleByType(inputs []float64, auxParams []float64, aType NodeActivationType) ([]float64, error) {
	if fn, ok := a.moduleActivators[aType]; ok {
		return fn(inputs, auxParams), nil
	} else {
		return nil, fmt.Errorf("unknown module activation type: %d", aType)
	}
}

// Register Registers given neuron activation function with provided type and name into the factory
func (a *NodeActivatorsFactory) Register(aType NodeActivationType, aFunc ActivationFunction, fName string) {
	// store function
	a.activators[aType] = aFunc
	// store name<->type bi-directional mapping
	a.forward[aType] = fName
	a.inverse[fName] = aType
}

// RegisterModule Registers given neuron module activation function with provided type and name into the factory
func (a *NodeActivatorsFactory) RegisterModule(aType NodeActivationType, aFunc ModuleActivationFunction, fName string) {
	// store function
	a.moduleActivators[aType] = aFunc
	// store name<->type bi-directional mapping
	a.forward[aType] = fName
	a.inverse[fName] = aType
}

// ActivationTypeFromName Parse node activation type name and return corresponding activation type
func (a *NodeActivatorsFactory) ActivationTypeFromName(name string) (NodeActivationType, error) {
	if t, ok := a.inverse[name]; ok {
		return t, nil
	} else {
		return math.MaxInt8, fmt.Errorf("unsupported activation type name: %s", name)
	}
}

// ActivationNameFromType Returns activation function name from given type
func (a *NodeActivatorsFactory) ActivationNameFromType(aType NodeActivationType) (string, error) {
	if n, ok := a.forward[aType]; ok {
		return n, nil
	} else {
		return "", fmt.Errorf("unsupported activation type: %d", aType)
	}
}

// The sigmoid activation functions
var (
	// The plain sigmoid
	plainSigmoid = func(input float64, auxParams []float64) float64 {
		return 1 / (1 + math.Exp(-input))
	}
	// The reduced sigmoid
	reducedSigmoid = func(input float64, auxParams []float64) float64 {
		return 1 / (1 + math.Exp(-0.5*input))
	}
	// The steepened sigmoid
	steepenedSigmoid = func(input float64, auxParams []float64) float64 {
		return 1.0 / (1.0 + math.Exp(-4.924273*input))
	}
	// The bipolar sigmoid activation function xrange->[-1,1] yrange->[-1,1]
	bipolarSigmoid = func(input float64, auxParams []float64) float64 {
		return (2.0 / (1.0 + math.Exp(-4.924273*input))) - 1.0
	}
	// The approximation sigmoid with squashing range [-4.0; 4.0]
	approximationSigmoid = func(input float64, auxParams []float64) float64 {
		four, one32nd := 4.0, 0.03125
		if input < -4.0 {
			return 0.0
		} else if input < 0.0 {
			return (input + four) * (input + four) * one32nd
		} else if input < 4.0 {
			return 1.0 - (input-four)*(input-four)*one32nd
		} else {
			return 1.0
		}
	}
	// The steepened approximation sigmoid with squashing range [-1.0; 1.0]
	approximationSteepenedSigmoid = func(input float64, auxParams []float64) float64 {
		one, oneHalf := 1.0, 0.5
		if input < -1.0 {
			return 0.0
		} else if input < 0.0 {
			return (input + one) * (input + one) * oneHalf
		} else if input < 1.0 {
			return 1.0 - (input-one)*(input-one)*oneHalf
		} else {
			return 1.0
		}
	}
	// The inverse absolute sigmoid
	inverseAbsoluteSigmoid = func(input float64, auxParams []float64) float64 {
		return 0.5 + (input/(1.0+math.Abs(input)))*0.5
	}

	// The left/right shifted sigmoid
	leftShiftedSigmoid = func(input float64, auxParams []float64) float64 {
		return 1.0 / (1.0 + math.Exp(-input-2.4621365))
	}
	leftShiftedSteepenedSigmoid = func(input float64, auxParams []float64) float64 {
		return 1.0 / (1.0 + math.Exp(-(4.924273*input + 2.4621365)))
	}
	rightShiftedSteepenedSigmoid = func(input float64, auxParams []float64) float64 {
		return 1.0 / (1.0 + math.Exp(-(4.924273*input - 2.4621365)))
	}
)

// The other activation functions
var (
	// The hyperbolic tangent
	hyperbolicTangent = func(input float64, auxParams []float64) float64 {
		return math.Tanh(0.9 * input)
	}
	// The bipolar Gaussian activator xrange->[-1,1] yrange->[-1,1]
	bipolarGaussian = func(input float64, auxParams []float64) float64 {
		return 2.0*math.Exp(-math.Pow(input*2.5, 2.0)) - 1.0
	}
	// The Gaussian activator xrange->[-1,1] yrange->[0,1]
	gaussian = func(input float64, auxParams []float64) float64 {
		return math.Exp(-math.Pow(input, 2.0))
	}
	// The absolute linear
	absoluteLinear = func(input float64, auxParams []float64) float64 {
		return math.Abs(input)
	}
	// Linear activation function with clipping. By 'clipping' we mean the output value is linear between
	/// x = -1 and x = 1. Below -1 and above +1 the output is clipped at -1 and +1 respectively
	clippedLinear = func(input float64, auxParams []float64) float64 {
		if input < -1.0 {
			return -1.0
		}
		if input > 1.0 {
			return 1.0
		}
		return input
	}
	// The linear activation
	linear = func(input float64, auxParams []float64) float64 {
		return input
	}
	// The null activator
	nullFunctor = func(input float64, auxParams []float64) float64 {
		return 0.0
	}
	// The sign activator
	signFunction = func(input float64, auxParams []float64) float64 {
		if math.IsNaN(input) || input == 0.0 {
			return 0.0
		} else if math.Signbit(input) {
			return -1.0
		} else {
			return 1.0
		}
	}
	// The sine periodic activation with doubled period
	sineFunction = func(input float64, auxParams []float64) float64 {
		return math.Sin(2.0 * input)
	}
	// The step function x<0 ? 0.0 : 1.0
	stepFunction = func(input float64, auxParams []float64) float64 {
		if math.Signbit(input) {
			return 0.0
		} else {
			return 1.0
		}
	}
)

// The modular activators
var (
	// Multiplies input values and returns multiplication result
	multiplyModule = func(inputs []float64, auxParams []float64) []float64 {
		ret := 1.0
		for _, v := range inputs {
			ret *= v
		}
		return []float64{ret}
	}
	// Finds maximal value among inputs and return it
	maxModule = func(inputs []float64, auxParams []float64) []float64 {
		maxVal := float64(math.MinInt64)
		for _, v := range inputs {
			maxVal = math.Max(maxVal, v)
		}
		return []float64{maxVal}
	}
	// Finds minimal value among inputs and returns it
	minModule = func(inputs []float64, auxParams []float64) []float64 {
		minVal := math.MaxFloat64
		for _, v := range inputs {
			minVal = math.Min(minVal, v)
		}
		return []float64{minVal}
	}
)
