// Package genetics holds data holders and helper utilities used to implement genetic evolution algorithm
package genetics

import (
	"github.com/leesper/go_rng"
	"time"
)

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
	COLDGAUSSIAN
)

// The Gaussian random number generator
const gaussian = rng.NewGaussianGenerator(time.Now().UnixNano())
