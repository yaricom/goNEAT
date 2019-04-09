package network

import (
	"fmt"
	"math"
	"errors"
	"github.com/yaricom/goNEAT/neat/utils"
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
	ActivationType utils.NodeActivationType
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
	activationFunctions         []utils.NodeActivationType
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
activationFunctions []utils.NodeActivationType, connections []*FastNetworkLink,
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
// Returns true if activation wave passed from all inputs to the outputs.
func (fmm *FastModularNetworkSolver) ForwardSteps(steps int) (res bool, err error) {
	for i := 0; i < steps; i++ {
		if res, err = fmm.forwardStep(0); err != nil {
			return false, err
		}
	}
	return res, nil
}

// Propagates activation wave through all network nodes provided number of steps by recursion from output nodes
// Returns true if activation wave passed from all inputs to the outputs. This method is preferred method
// of network activation when number of forward steps can not be easy calculated and no network modules are set.
func (fmm *FastModularNetworkSolver) RecursiveSteps() (res bool, err error) {
	if len(fmm.modules) > 0 {
		return false, errors.New("recursive activation can not be used for network with defined modules")
	}

	// Initialize boolean arrays and set the last activation signal for output/hidden neurons
	for i := 0; i < fmm.totalNeuronCount; i++ {
		// Set as activated if i is an input node, otherwise ensure it is unactivated (false)
		fmm.activated[i] = i < fmm.sensorNeuronCount

		fmm.inActivation[i] = false
		// set last activation for output/hidden neurons
		if i >= fmm.sensorNeuronCount {
			fmm.lastActivation[i] = fmm.neuronSignals[i]
		}
	}

	// Get each output node activation recursively
	for i := 0; i < fmm.outputNeuronCount; i++ {
		if res, err = fmm.recursiveActivateNode(fmm.sensorNeuronCount + i); err != nil {
			return false, err
		}
	}

	return true, nil
}

// Propagate activation wave by recursively looking for input signals graph for a given output neuron
func (fmm *FastModularNetworkSolver) recursiveActivateNode(currentNode int) (res bool, err error) {
	// If we've reached an input node then return since the signal is already set
	if fmm.activated[currentNode] {
		fmm.inActivation[currentNode] = false
		return true, nil
	}
	// Mark that the node is currently being calculated
	fmm.inActivation[currentNode] = true

	// Set the pre-signal to 0
	fmm.neuronSignalsBeingProcessed[currentNode] = 0

	// Adjacency list in reverse holds incoming connections, go through each one and activate it
	for i := 0; i < len(fmm.reverseAdjacentList[currentNode]); i++ {
		crntAdjNode := fmm.reverseAdjacentList[currentNode][i]

		// If this node is currently being activated then we have reached a cycle, or recurrent connection.
		// Use the previous activation in this case
		if fmm.inActivation[crntAdjNode] {
			fmm.neuronSignalsBeingProcessed[currentNode] += fmm.lastActivation[crntAdjNode] * fmm.adjacentMatrix[crntAdjNode][currentNode]
		} else {
			// Otherwise proceed as normal
			// Recurse if this neuron has not been activated yet
			if !fmm.activated[crntAdjNode] {
				res, err = fmm.recursiveActivateNode(crntAdjNode)
				if err != nil {
					// recursive activation failed
					return false, err
				}
			}

			// Add it to the new activation
			fmm.neuronSignalsBeingProcessed[currentNode] += fmm.neuronSignals[crntAdjNode] * fmm.adjacentMatrix[crntAdjNode][currentNode]
		}
	}

	// Mark this neuron as completed
	fmm.activated[currentNode] = true

	// This is no longer being calculated (for cycle detection)
	fmm.inActivation[currentNode] = false

	// Set this signal after running it through the activation function
	if fmm.neuronSignals[currentNode], err = utils.NodeActivators.ActivateByType(
		fmm.neuronSignalsBeingProcessed[currentNode], nil,
		fmm.activationFunctions[currentNode]); err != nil {
		// failed to activate
		res = false
	}
	return res, err
}

// Attempts to relax network given amount of steps until giving up. The network considered relaxed when absolute
// value of the change at any given point is less than maxAllowedSignalDelta during activation waves propagation.
// If maxAllowedSignalDelta value is less than or equal to 0, the method will return true without checking for relaxation.
func (fmm *FastModularNetworkSolver) Relax(maxSteps int, maxAllowedSignalDelta float64) (relaxed bool, err error) {
	for i := 0; i < maxSteps; i++ {
		if relaxed, err = fmm.forwardStep(maxAllowedSignalDelta); err != nil {
			return false, err
		} else if relaxed {
			break // no need to iterate any further, already reached desired accuracy
		}
	}
	return relaxed, nil
}

// Performs single forward step through the network and tests if network become relaxed. The network considered relaxed
// when absolute value of the change at any given point is less than maxAllowedSignalDelta during activation waves propagation.
func (fmm *FastModularNetworkSolver) forwardStep(maxAllowedSignalDelta float64) (isRelaxed bool, err error) {
	isRelaxed = true

	// Calculate output signal per each connection and add the signals to the target neurons
	for _, conn := range fmm.connections {
		fmm.neuronSignalsBeingProcessed[conn.TargetIndx] += fmm.neuronSignals[conn.SourceIndx] * conn.Weight
	}

	// Pass the signals through the single-valued activation functions
	for i := fmm.sensorNeuronCount; i < fmm.totalNeuronCount; i++ {
		signal := fmm.neuronSignalsBeingProcessed[i]
		if fmm.biasNeuronCount > 0 {
			// append BIAS value to the signal if appropriate
			signal += fmm.biasList[i]
		}

		if fmm.neuronSignalsBeingProcessed[i], err = utils.NodeActivators.ActivateByType(
			signal, nil, fmm.activationFunctions[i]); err != nil {
			return false, err
		}
	}

	// Pass the signals through each module (activation function with more than one input or output)
	for _, module := range fmm.modules {
		inputs := make([]float64, len(module.InputIndxs))
		for i, in_index := range module.InputIndxs {
			inputs[i] = fmm.neuronSignalsBeingProcessed[in_index]
		}
		if outputs, err := utils.NodeActivators.ActivateModuleByType(inputs, nil, module.ActivationType); err == nil {
			// save outputs
			for i, out_index := range module.OutputIndxs {
				fmm.neuronSignalsBeingProcessed[out_index] = outputs[i]
			}
		} else {
			return false, err
		}
	}

	// Move all the neuron signals we changed while processing this network activation into storage.
	if maxAllowedSignalDelta <= 0 {
		// iterate through output and hidden neurons and collect activations
		for i := fmm.sensorNeuronCount; i < fmm.totalNeuronCount; i++ {
			fmm.neuronSignals[i] = fmm.neuronSignalsBeingProcessed[i]
			fmm.neuronSignalsBeingProcessed[i] = 0
		}
	} else {
		for i := fmm.sensorNeuronCount; i < fmm.totalNeuronCount; i++ {
			// First check whether any location in the network has changed by more than a small amount.
			isRelaxed = isRelaxed && !(math.Abs(fmm.neuronSignals[i] - fmm.neuronSignalsBeingProcessed[i]) > maxAllowedSignalDelta)

			fmm.neuronSignals[i] = fmm.neuronSignalsBeingProcessed[i]
			fmm.neuronSignalsBeingProcessed[i] = 0
		}
	}

	return isRelaxed, err
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
		// only inputs should be provided
		for i := 0; i < fmm.inputNeuronCount; i++ {
			fmm.neuronSignals[fmm.biasNeuronCount + i] = inputs[i]
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
	return fmm.totalNeuronCount + len(fmm.modules)
}
// Returns the total number of links between nodes in the network
func (fmm *FastModularNetworkSolver) LinkCount() int {
	// count all connections
	num_links := len(fmm.connections)

	// count all bias links if any
	if fmm.biasNeuronCount > 0 {
		for _, b := range fmm.biasList {
			if b != 0 {
				num_links++
			}
		}
	}

	// count all modules links
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
