// The package network provides data holders and utilities to describe Artificial Neural Network
package network

// These are NNode types
const (
	// The neuron type
	NEURON = iota
	// The sensor type
	SENSOR
)

// These are NNode layer type
const (
	// The node is in hidden layer
	HIDDEN = iota
	// The node is in input layer
	INPUT
	// The node is in output layer
	OUTPUT
)


// Mutators are variables that specify a kind of mutation of connection weights between NN nodes
const (
	//This adds Gaussian noise to the weights
	GAUSSIAN = iota
	//This sets weights to numbers chosen from a Gaussian distribution
	COLDGAUSSIAN
)

// The neuron Activation function Types
const (
	// The sigmoid activation function
	SIGMOID = iota
)

// The innovation method to be applied
const (
	// The novelty will be introduced by new NN node
	NEWNODE
	// The novelty will be introduced by new NN link
	NEWLINK
)