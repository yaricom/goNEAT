// Package genetics holds data holders and helper utilities used to implement genetic evolution algorithm
package genetics

import "errors"

// The innovation method type to be applied
type innovationType byte
// Available innovation types
const (
	// The novelty will be introduced by new NN node
	newNodeInnType innovationType = iota + 1
	// The novelty will be introduced by new NN link
	newLinkInnType
)


// The mutator type that specifies a kind of mutation of connection weights between NN nodes
type mutatorType byte
// Available mutator types
const (
	//This adds Gaussian noise to the weights
	gaussianMutator mutatorType = iota + 1
	//This sets weights to numbers chosen from a Gaussian distribution
	goldGaussianMutator
)

// Defines format of Genome data encoding
type GenomeEncoding byte

const (
	// The plain text
	PlainGenomeEncoding GenomeEncoding = iota + 1
	// The rich text in YAML
	YAMLGenomeEncoding
)

var (
	ErrUnsupportedGenomeEncoding = errors.New("unsupported genome encoding")
)

