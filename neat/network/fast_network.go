package network

import (
	"fmt"
	"errors"
)

// The connection descriptor for fast network
type FastNetworkLink struct {
	// The index of source neuron
	SourceIndx int
	// The index of target neuron
	TargetIndx int
	// The weight of this link
	Weight     float64
	// The signal relayed by this link
	Signal     float64
}
// The module relay (control node) descriptor for fast network
type FastControlNode struct {
	// The activation function for control node
	ActivationType NodeActivationType
	// The indexes of the input nodes
	InputIndxs     []int
	// The indexes of the output nodes
	OutputIndxs    []int
}

// The fast modular network solver implementation to be used for big neural networks simulation.
type FastModularNetworkSolver struct {
	// A network id
	Id                          int
	// Is a name of this network */
	Name                        string

	// The current activation values per each neuron
	neuronSignals               []float64
	// This array is a parallel of neuronSignals and used to test network relaxation
	neuronSignalsBeingProcessed []float64

	// The activation functions per neuron, must be in the same order as neuronSignals. Has nil entries for
	// neurons that are inputs or outputs of a module.
	activationFunctions         []NodeActivationType
	// The bias values associated with neurons
	biasList                    []float64
	// The control nodes relaying between network modules
	modules                     []*FastControlNode
	// The connections
	connections                 []*FastNetworkLink

	// The number of input neurons
	inputNeuronCount            int
	// The total number of sensors in the network (input + bias). This is also the index of the first output neuron in the neuron signals.
	sensorNeuronCount           int
	// The number of output neurons
	outputNeuronCount           int
	// The bias neuron count (usually one). This is also the index of the first input neuron in the neuron signals.
	biasNeuronCount             int
	// The total number of neurons in network
	totalNeuronCount            int

	// For recursive activation, marks whether we have finished this node yet
	activated                   []bool
	// For recursive activation, makes whether a node is currently being calculated (recurrent connections processing)
	inActivation                []bool
	// For recursive activation, the previous activation values of recurrent connections (recurrent connections processing)
	lastActivation              []float64

	// The adjacent list to hold IDs of outgoing nodes for each network node
	adjacentList                [][]int
	// The adjacent list to hold IDs of incoming nodes for each network node
	reverseAdjacentList         [][]int
	// The adjacent matrix to hold connection weights between all connected nodes
	adjacentMatrix              [][]float64
}

// Creates new fast modular network solver
func NewFastModularNetworkSolver(biasNeuronCount, inputNeuronCount, outputNeuronCount, totalNeuronCount int,
activationFunctions []NodeActivationType, connections []*FastNetworkLink,
biasList []float64, modules []*FastControlNode) *FastModularNetworkSolver {

	fmm := FastModularNetworkSolver{
		biasNeuronCount:biasNeuronCount,
		inputNeuronCount:inputNeuronCount,
		sensorNeuronCount:biasNeuronCount + inputNeuronCount,
		outputNeuronCount:outputNeuronCount,
		totalNeuronCount:totalNeuronCount,
		activationFunctions:activationFunctions,
		biasList:biasList,
		modules:modules,
		connections:connections,
	}

	// Allocate the arrays that store the states at different points in the neural network.
	// The neuron signals are initialised to 0 by default. Only bias nodes need setting to 1.
	fmm.neuronSignals = make([]float64, totalNeuronCount)
	fmm.neuronSignalsBeingProcessed = make([]float64, totalNeuronCount)
	for i := 0; i < biasNeuronCount; i++ {
		fmm.neuronSignals[i] = 1.0 // BIAS neuron signal
	}

	// Allocate activation arrays
	fmm.activated = make([]bool, totalNeuronCount)
	fmm.inActivation = make([]bool, totalNeuronCount)
	fmm.lastActivation = make([]float64, totalNeuronCount)

	// Build adjacent lists and matrix for fast access of incoming/outgoing nodes and connection weights
	fmm.adjacentList = make([][]int, totalNeuronCount)
	fmm.reverseAdjacentList = make([][]int, totalNeuronCount)
	fmm.adjacentMatrix = make([][]float64, totalNeuronCount)

	for i := 0; i < totalNeuronCount; i++ {
		fmm.adjacentList[i] = make([]int, 0)
		fmm.reverseAdjacentList[i] = make([]int, 0)
		fmm.adjacentMatrix[i] = make([]float64, totalNeuronCount)
	}

	for i := 0; i < len(connections); i++ {
		crs := connections[i].SourceIndx
		crt := connections[i].TargetIndx
		// Holds outgoing nodes
		fmm.adjacentList[crs] = append(fmm.adjacentList[crs], crt)
		// Holds incoming nodes
		fmm.reverseAdjacentList[crt] = append(fmm.reverseAdjacentList[crt], crs)
		// Holds link weight
		fmm.adjacentMatrix[crs][crt] = connections[i].Weight
	}

	return &fmm
}

// Propagates activation wave through all network nodes provided number of steps in forward direction.
// Returns true if activation wave passed from all inputs to outputs.
func (fmm *FastModularNetworkSolver) ForwardSteps(steps int) (bool, error) {
	return false, errors.New("not implemented")
}

// Propagates activation wave through all network nodes provided number of steps by recursion from output nodes
// Returns true if activation wave passed from all inputs to outputs.
func (fmm *FastModularNetworkSolver) RecursiveSteps() (bool, error) {
	return false, errors.New("not implemented")
}

// Attempts to relax network given amount of steps until giving up. The network considered relaxed when absolute
// value of the change at any given point is less than maxAllowedSignalDelta during activation waves propagation.
// If maxAllowedSignalDelta value is less than or equal to 0, the method will return true without checking for relaxation.
func (fmm *FastModularNetworkSolver) Relax(maxSteps int, maxAllowedSignalDelta float64) (bool, error) {
	return false, errors.New("not implemented")
}

// Flushes network state by removing all current activations. Returns true if network flushed successfully or
// false in case of error.
func (fmm *FastModularNetworkSolver) Flush() (bool, error) {
	for i := fmm.biasNeuronCount; i < fmm.totalNeuronCount; i++ {
		fmm.neuronSignals[i] = 0.0
	}
	return true, nil
}

// Set sensors values to the input nodes of the network
func (fmm *FastModularNetworkSolver) LoadSensors(inputs []float64) error {
	if len(inputs) == fmm.inputNeuronCount {
		// only inputs provided
		for i := 0; i < fmm.inputNeuronCount; i++ {
			fmm.neuronSignals[fmm.biasNeuronCount + i] = inputs[i]
		}
	} else if len(inputs) == fmm.sensorNeuronCount  {
		// inputs and bias provided
		for i := 0; i < fmm.sensorNeuronCount; i++ {
			fmm.neuronSignals[i] = inputs[i]
		}
	} else {
		return NetErrUnsupportedSensorsArraySize
	}
	return nil
}

// Read output values from the output nodes of the network
func (fmm *FastModularNetworkSolver) ReadOutputs() []float64 {
	return fmm.neuronSignals[fmm.sensorNeuronCount:fmm.sensorNeuronCount + fmm.outputNeuronCount]
}

// Returns the total number of neural units in the network
func (fmm *FastModularNetworkSolver) NodeCount() int {
	return fmm.totalNeuronCount
}
// Returns the total number of links between nodes in the network
func (fmm *FastModularNetworkSolver) LinkCount() int {
	num_links := len(fmm.connections)
	if len(fmm.modules) != 0 {
		for _, module := range fmm.modules {
			num_links += len(module.InputIndxs) + len(module.OutputIndxs)
		}
	}
	return num_links
}

// Stringer
func (fmm *FastModularNetworkSolver) String() string {
	str := fmt.Sprintf("FastModularNetwork, id: %d, name: [%s], neurons: %d,\n\tinputs: %d,\tbias: %d,\toutputs:%d,\t hidden: %d",
		fmm.Id, fmm.Name, fmm.totalNeuronCount, fmm.inputNeuronCount, fmm.biasNeuronCount, fmm.outputNeuronCount,
		fmm.totalNeuronCount - fmm.sensorNeuronCount - fmm.outputNeuronCount)
	return str
}
