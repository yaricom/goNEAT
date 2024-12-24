package network

import (
	"errors"
	"fmt"
	neatmath "github.com/yaricom/goNEAT/v4/neat/math"
	"math"
)

// FastNetworkLink The connection descriptor for fast network
type FastNetworkLink struct {
	// The index of source neuron
	SourceIndex int `json:"source_index"`
	// The index of target neuron
	TargetIndex int `json:"target_index"`
	// The weight of this link
	Weight float64 `json:"weight"`
	// The signal relayed by this link
	Signal float64 `json:"signal"`
}

// FastControlNode The module relay (control node) descriptor for fast network
type FastControlNode struct {
	// The activation function for control node
	ActivationType neatmath.NodeActivationType
	// The indexes of the input nodes
	InputIndexes []int
	// The indexes of the output nodes
	OutputIndexes []int
}

// FastModularNetworkSolver is the network solver implementation to be used for large neural networks simulation.
type FastModularNetworkSolver struct {
	// A network id
	Id int
	// Is a name of this network */
	Name string

	// The current activation values per each neuron
	neuronSignals []float64
	// This array is a parallel of neuronSignals and used to test network relaxation
	neuronSignalsBeingProcessed []float64

	// The activation functions per neuron, must be in the same order as neuronSignals. Has nil entries for
	// neurons that are inputs or outputs of a module.
	activationFunctions []neatmath.NodeActivationType
	// The bias values associated with neurons
	biasList []float64
	// The control nodes relaying between network modules
	modules []*FastControlNode
	// The connections
	connections []*FastNetworkLink

	// The number of input neurons
	inputNeuronCount int
	// The total number of sensors in the network (input + bias). This is also the index of the first output neuron in the neuron signals.
	sensorNeuronCount int
	// The number of output neurons
	outputNeuronCount int
	// The bias neuron count (usually one). This is also the index of the first input neuron in the neuron signals.
	biasNeuronCount int
	// The total number of neurons in network
	totalNeuronCount int

	// For recursive activation, marks whether we have finished this node yet
	activated []bool
	// For recursive activation, makes whether a node is currently being calculated (recurrent connections processing)
	inActivation []bool
	// For recursive activation, the previous activation values of recurrent connections (recurrent connections processing)
	lastActivation []float64

	// The adjacency list to hold IDs of incoming nodes for each network node
	reverseAdjacencyList [][]int
	// The adjacency matrix to hold connection weights between all connected nodes
	adjacencyMatrix [][]float64
}

// NewFastModularNetworkSolver Creates new fast modular network solver
func NewFastModularNetworkSolver(biasNeuronCount, inputNeuronCount, outputNeuronCount, totalNeuronCount int,
	activationFunctions []neatmath.NodeActivationType, connections []*FastNetworkLink,
	biasList []float64, modules []*FastControlNode) *FastModularNetworkSolver {

	fmm := FastModularNetworkSolver{
		biasNeuronCount:     biasNeuronCount,
		inputNeuronCount:    inputNeuronCount,
		sensorNeuronCount:   biasNeuronCount + inputNeuronCount,
		outputNeuronCount:   outputNeuronCount,
		totalNeuronCount:    totalNeuronCount,
		activationFunctions: activationFunctions,
		biasList:            biasList,
		modules:             modules,
		connections:         connections,
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
	fmm.reverseAdjacencyList = make([][]int, totalNeuronCount)
	fmm.adjacencyMatrix = make([][]float64, totalNeuronCount)

	for i := 0; i < totalNeuronCount; i++ {
		fmm.reverseAdjacencyList[i] = make([]int, 0)
		fmm.adjacencyMatrix[i] = make([]float64, totalNeuronCount)
	}

	for i := 0; i < len(connections); i++ {
		crs := connections[i].SourceIndex
		crt := connections[i].TargetIndex
		// Holds incoming nodes
		fmm.reverseAdjacencyList[crt] = append(fmm.reverseAdjacencyList[crt], crs)
		// Holds link weight
		fmm.adjacencyMatrix[crs][crt] = connections[i].Weight
	}

	return &fmm
}

func (s *FastModularNetworkSolver) ForwardSteps(steps int) (res bool, err error) {
	for i := 0; i < steps; i++ {
		if res, err = s.forwardStep(0); err != nil {
			return false, err
		}
	}
	return res, nil
}

func (s *FastModularNetworkSolver) RecursiveSteps() (res bool, err error) {
	if len(s.modules) > 0 {
		return false, errors.New("recursive activation can not be used for network with defined modules")
	}

	// Initialize boolean arrays and set the last activation signal for output/hidden neurons
	for i := 0; i < s.totalNeuronCount; i++ {
		// Set as activated if i is an input node, otherwise ensure it is unactivated (false)
		s.activated[i] = i < s.sensorNeuronCount

		s.inActivation[i] = false
		// set last activation for output/hidden neurons
		if i >= s.sensorNeuronCount {
			s.lastActivation[i] = s.neuronSignals[i]
		}
	}

	// Get each output node activation recursively
	for i := 0; i < s.outputNeuronCount; i++ {
		index := s.sensorNeuronCount + i
		if res, err = s.recursiveActivateNode(index); err != nil {
			return false, err
		} else if !res {
			return false, fmt.Errorf("failed to recursively activate the output neuron at %d", index)
		}
	}

	return res, nil
}

// Propagate activation wave by recursively looking for input signals graph for a given output neuron
func (s *FastModularNetworkSolver) recursiveActivateNode(currentNode int) (res bool, err error) {
	// If we've reached an input node then return since the signal is already set
	if s.activated[currentNode] {
		s.inActivation[currentNode] = false
		return true, nil
	}
	// Mark that the node is currently being calculated
	s.inActivation[currentNode] = true

	// Set the pre-signal to 0
	s.neuronSignalsBeingProcessed[currentNode] = 0

	// Adjacency list in reverse holds incoming connections, go through each one and activate it
	for i := 0; i < len(s.reverseAdjacencyList[currentNode]); i++ {
		currentAdjNode := s.reverseAdjacencyList[currentNode][i]

		// If this node is currently being activated then we have reached a cycle, or recurrent connection.
		// Use the previous activation in this case
		if s.inActivation[currentAdjNode] {
			s.neuronSignalsBeingProcessed[currentNode] += s.lastActivation[currentAdjNode] * s.adjacencyMatrix[currentAdjNode][currentNode]
		} else {
			// Otherwise, proceed as normal
			// Recurse if this neuron has not been activated yet
			if !s.activated[currentAdjNode] {
				res, err = s.recursiveActivateNode(currentAdjNode)
				if err != nil {
					// recursive activation failed
					return false, err
				} else if !res {
					return false, fmt.Errorf("failed to recursively activate neuron at %d", currentAdjNode)
				}
			}

			// Add it to the new activation
			s.neuronSignalsBeingProcessed[currentNode] += s.neuronSignals[currentAdjNode] * s.adjacencyMatrix[currentAdjNode][currentNode]
		}
	}

	// Mark this neuron as completed
	s.activated[currentNode] = true

	// This is no longer being calculated (for cycle detection)
	s.inActivation[currentNode] = false

	// Set this signal after running it through the activation function
	if s.neuronSignals[currentNode], err = neatmath.NodeActivators.ActivateByType(
		s.neuronSignalsBeingProcessed[currentNode], nil,
		s.activationFunctions[currentNode]); err != nil {
		// failed to activate
		res = false
	} else {
		res = true
	}
	return res, err
}

func (s *FastModularNetworkSolver) Relax(maxSteps int, maxAllowedSignalDelta float64) (relaxed bool, err error) {
	for i := 0; i < maxSteps; i++ {
		if relaxed, err = s.forwardStep(maxAllowedSignalDelta); err != nil {
			return false, err
		} else if relaxed {
			break // no need to iterate any further, already reached desired accuracy
		}
	}
	return relaxed, nil
}

// Performs single forward step through the network and tests if network become relaxed. The network considered relaxed
// when absolute value of the change at any given point is less than maxAllowedSignalDelta during activation waves propagation.
func (s *FastModularNetworkSolver) forwardStep(maxAllowedSignalDelta float64) (isRelaxed bool, err error) {
	isRelaxed = true

	// Calculate output signal per each connection and add the signals to the target neurons
	for _, conn := range s.connections {
		s.neuronSignalsBeingProcessed[conn.TargetIndex] += s.neuronSignals[conn.SourceIndex] * conn.Weight
	}

	// Pass the signals through the single-valued activation functions
	for i := s.sensorNeuronCount; i < s.totalNeuronCount; i++ {
		signal := s.neuronSignalsBeingProcessed[i]
		if s.biasNeuronCount > 0 {
			// append BIAS value to the signal if appropriate
			signal += s.biasList[i]
		}

		if s.neuronSignalsBeingProcessed[i], err = neatmath.NodeActivators.ActivateByType(
			signal, nil, s.activationFunctions[i]); err != nil {
			return false, err
		}
	}

	// Pass the signals through each module (activation function with more than one input or output)
	for _, module := range s.modules {
		inputs := make([]float64, len(module.InputIndexes))
		for i, inIndex := range module.InputIndexes {
			inputs[i] = s.neuronSignalsBeingProcessed[inIndex]
		}
		if outputs, err := neatmath.NodeActivators.ActivateModuleByType(inputs, nil, module.ActivationType); err == nil {
			// save outputs
			for i, outIndex := range module.OutputIndexes {
				s.neuronSignalsBeingProcessed[outIndex] = outputs[i]
			}
		} else {
			return false, err
		}
	}

	// Move all the neuron signals we changed while processing this network activation into storage.
	if maxAllowedSignalDelta <= 0 {
		// iterate through output and hidden neurons and collect activations
		for i := s.sensorNeuronCount; i < s.totalNeuronCount; i++ {
			s.neuronSignals[i] = s.neuronSignalsBeingProcessed[i]
			s.neuronSignalsBeingProcessed[i] = 0
		}
	} else {
		for i := s.sensorNeuronCount; i < s.totalNeuronCount; i++ {
			// First check whether any location in the network has changed by more than a small amount.
			isRelaxed = isRelaxed && !(math.Abs(s.neuronSignals[i]-s.neuronSignalsBeingProcessed[i]) > maxAllowedSignalDelta)

			s.neuronSignals[i] = s.neuronSignalsBeingProcessed[i]
			s.neuronSignalsBeingProcessed[i] = 0
		}
	}

	return isRelaxed, err
}

func (s *FastModularNetworkSolver) Flush() (bool, error) {
	for i := s.biasNeuronCount; i < s.totalNeuronCount; i++ {
		s.neuronSignals[i] = 0.0
		s.neuronSignalsBeingProcessed[i] = 0.0
	}
	return true, nil
}

func (s *FastModularNetworkSolver) LoadSensors(inputs []float64) error {
	if len(inputs) == s.inputNeuronCount {
		// only inputs should be provided
		for i := 0; i < s.inputNeuronCount; i++ {
			s.neuronSignals[s.biasNeuronCount+i] = inputs[i]
		}
	} else {
		return ErrNetUnsupportedSensorsArraySize
	}
	return nil
}

func (s *FastModularNetworkSolver) ReadOutputs() []float64 {
	// decouple and return
	outs := make([]float64, s.outputNeuronCount)
	copy(outs, s.neuronSignals[s.sensorNeuronCount:s.sensorNeuronCount+s.outputNeuronCount])
	return outs
}

func (s *FastModularNetworkSolver) NodeCount() int {
	return s.totalNeuronCount + len(s.modules)
}

func (s *FastModularNetworkSolver) LinkCount() int {
	// count all connections
	numLinks := len(s.connections)

	// count all bias links if any
	if s.biasNeuronCount > 0 {
		for _, b := range s.biasList {
			if b != 0 {
				numLinks++
			}
		}
	}

	// count all modules links
	if len(s.modules) != 0 {
		for _, module := range s.modules {
			numLinks += len(module.InputIndexes) + len(module.OutputIndexes)
		}
	}
	return numLinks
}

// Stringer
func (s *FastModularNetworkSolver) String() string {
	str := fmt.Sprintf("FastModularNetwork, id: %d, name: [%s], neurons: %d,\n\tinputs: %d,\tbias: %d,\toutputs:%d,\t hidden: %d",
		s.Id, s.Name, s.totalNeuronCount, s.inputNeuronCount, s.biasNeuronCount, s.outputNeuronCount,
		s.totalNeuronCount-s.sensorNeuronCount-s.outputNeuronCount)
	return str
}
