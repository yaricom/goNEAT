// Package genetics holds data holders and helper utilities used to implement genetic evolution algorithm
package genetics

import (
	"errors"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/network"
)

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

// GenomeEncoding Defines format of Genome data encoding
type GenomeEncoding byte

const (
	// PlainGenomeEncoding The plain text
	PlainGenomeEncoding GenomeEncoding = iota + 1
	// YAMLGenomeEncoding The rich text in YAML
	YAMLGenomeEncoding
)

var (
	ErrUnsupportedGenomeEncoding = errors.New("unsupported genome encoding")
)

// TraitWithId Utility to select trait with given ID from provided Traits array
func TraitWithId(traitId int, traits []*neat.Trait) *neat.Trait {
	if traitId != 0 && traits != nil {
		for _, tr := range traits {
			if tr.Id == traitId {
				return tr
			}
		}
	}
	return nil
}

// NodeWithId Utility to select NNode with given ID from provided NNodes array
func NodeWithId(nodeId int, nodes []*network.NNode) *network.NNode {
	if nodeId != 0 && nodes != nil {
		for _, n := range nodes {
			if n.Id == nodeId {
				return n
			}
		}
	}
	return nil
}
