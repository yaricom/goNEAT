package network

// The fast modular network solver implementation to be used for big neural networks simulation.
type FastModularNetwork struct {
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
	activationFunctions         []ActivationFunction

	// The number of input neurons
	inputNeuronCount            int
	// The total number of sensors in the network (input + bias). This is also the index of the first output neuron in the neuron signals.
	sensorNeuronCount           int
	// The number of output neurons
	outputNeuronCount           int
	// The bias neuron count (usually one). This is also the index of the first input neuron in the neuron signals.
	biasNeuronCount             int

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
