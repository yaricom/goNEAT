// Package genetics holds data holders and helper utilities used to implement genetic evolution algorithm
package genetics

// The innovation method to be applied
const (
	// The novelty will be introduced by new NN node
	NEWNODE = iota
	// The novelty will be introduced by new NN link
	NEWLINK
)

