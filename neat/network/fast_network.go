package network

import "fmt"

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
func NewFastModularNetwork(biasNeuronCount, inputNeuronCount, outputNeuronCount, totalNeuronCount int,
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
	}

	// TODO! Continue initialization

	return &fmm
}

// Stringer
func (fmm *FastModularNetworkSolver) String() string {
	str := fmt.Sprintf("FastModularNetwork, id: %d, name: [%s], neurons: %d,\n\tinputs: %d,\tbias: %d,\toutputs:%d,\t hidden: %d",
		fmm.Id, fmm.Name, fmm.totalNeuronCount, fmm.inputNeuronCount, fmm.biasNeuronCount, fmm.outputNeuronCount,
		fmm.totalNeuronCount - fmm.sensorNeuronCount - fmm.outputNeuronCount)
	return str
}
