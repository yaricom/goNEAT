package network

import (
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
)

// Link is a connection from one node to another with an associated weight.
// It can be marked as recurrent.
type Link struct {
	// weight of connection
	ConnectionWeight float64
	// NNode inputting into the link
	InNode *NNode
	// NNode that the link affects
	OutNode *NNode
	// If TRUE the link is recurrent
	IsRecurrent bool
	// If TRUE the link is time delayed
	IsTimeDelayed bool

	// Points to a trait of parameters for genetic creation
	Trait *neat.Trait

	/* ************ LEARNING PARAMETERS *********** */
	// The following parameters are for use in neurons that learn through habituation,
	// sensitization, or Hebbian-type processes
	Params []float64
}

// NewLink Creates new link with specified weight, input and output neurons connected recurrently or not.
func NewLink(weight float64, inputNode, outNode *NNode, recurrent bool) *Link {
	link := newLink(weight)
	link.InNode = inputNode
	link.OutNode = outNode
	link.IsRecurrent = recurrent
	return link
}

// NewLinkWithTrait Creates new Link with specified Trait
func NewLinkWithTrait(trait *neat.Trait, weight float64, inputNode, outNode *NNode, recurrent bool) *Link {
	link := newLink(weight)
	link.InNode = inputNode
	link.OutNode = outNode
	link.IsRecurrent = recurrent
	link.Trait = trait
	link.deriveTrait(trait)
	return link
}

// NewLinkCopy The copy constructor to create new link with parameters taken from provided ones and connecting specified nodes
func NewLinkCopy(l *Link, inputNode, outNode *NNode) *Link {
	link := newLink(l.ConnectionWeight)
	link.InNode = inputNode
	link.OutNode = outNode
	link.Trait = l.Trait
	link.deriveTrait(l.Trait)
	link.IsRecurrent = l.IsRecurrent
	return link
}

// The private default constructor
func newLink(weight float64) *Link {
	return &Link{
		ConnectionWeight: weight,
	}
}

// IsEqualGenetically Checks if this link is genetically equal to provided one, i.e. connects nodes with the same IDs and has equal
// recurrent flag. I.e. if both links represent the same Gene.
func (l *Link) IsEqualGenetically(ol *Link) bool {
	sameInNode := l.InNode.Id == ol.InNode.Id
	sameOutNode := l.OutNode.Id == ol.OutNode.Id
	sameRecurrent := l.IsRecurrent == ol.IsRecurrent

	return sameInNode && sameOutNode && sameRecurrent
}

// The Link methods implementation
func (l *Link) String() string {
	return fmt.Sprintf("[Link: (%s <-> %s), weight: %.3f, recurrent: %t, time delayed: %t]",
		l.InNode, l.OutNode, l.ConnectionWeight, l.IsRecurrent, l.IsTimeDelayed)
}

// IDString is to get synthetic ID of this link composed of IDs of connected nodes.
func (l *Link) IDString() string {
	return fmt.Sprintf("%d-%d", l.InNode.Id, l.OutNode.Id)
}

// Copy trait parameters into this link's parameters
func (l *Link) deriveTrait(t *neat.Trait) {
	if t != nil {
		l.Params = make([]float64, len(t.Params))
		for i, p := range t.Params {
			l.Params[i] = p
		}
	}
}
