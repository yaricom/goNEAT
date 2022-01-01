package network

// Solver defines network solver interface, which allows propagation of the activation waves through the underlying network graph.
type Solver interface {
	// ForwardSteps Propagates activation wave through all network nodes provided number of steps in forward direction.
	// Normally the number of steps should be equal to the activation depth of the network.
	// Returns true if activation wave passed from all inputs to the output nodes.
	ForwardSteps(steps int) (bool, error)

	// RecursiveSteps Propagates activation wave through all network nodes provided number of steps by recursion from output nodes
	// Returns true if activation wave passed from all inputs to the output nodes.
	RecursiveSteps() (bool, error)

	// Relax Attempts to relax network given amount of steps until giving up. The network considered relaxed when absolute
	// value of the change at any given point is less than maxAllowedSignalDelta during activation waves propagation.
	// If maxAllowedSignalDelta value is less than or equal to 0, the method will return true without checking for relaxation.
	Relax(maxSteps int, maxAllowedSignalDelta float64) (bool, error)

	// Flush Flushes network state by removing all current activations. Returns true if network flushed successfully or
	// false in case of error.
	Flush() (bool, error)

	// LoadSensors Set sensors values to the input nodes of the network
	LoadSensors(inputs []float64) error
	// ReadOutputs Read output values from the output nodes of the network
	ReadOutputs() []float64

	// NodeCount Returns the total number of neural units in the network
	NodeCount() int
	// LinkCount Returns the total number of links between nodes in the network
	LinkCount() int
}
