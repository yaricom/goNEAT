// Package genetics holds data holders and helper utilities used to implement genetic evolution algorithm
package genetics

// The innovation method to be applied
const (
	// The novelty will be introduced by new NN node
	NEWNODE = iota
	// The novelty will be introduced by new NN link
	NEWLINK
)

// Mutators are variables that specify a kind of mutation of connection weights between NN nodes
const (
	//This adds Gaussian noise to the weights
	GAUSSIAN = iota
	//This sets weights to numbers chosen from a Gaussian distribution
	COLD_GAUSSIAN
)

