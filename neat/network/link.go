package network

import (
	"fmt"
	"github.com/yaricom/goNEAT/neat"
)

// A LINK is a connection from one node to another with an associated weight.
// It can be marked as recurrent.
type Link struct {
	// Weight of connection
	Weight        float64
	// NNode inputting into the link
	InNode        *NNode
	// NNode that the link affects
	OutNode       *NNode
	// If TRUE the link is recurrent
	IsRecurrent   bool
	// If TRUE the link is time delayed
	IsTimeDelayed bool

	// Points to a trait of parameters for genetic creation
	Trait         *neat.Trait

	/* ************ LEARNING PARAMETERS *********** */
	// The following parameters are for use in neurons that learn through habituation,
	// sensitization, or Hebbian-type processes
	Params        []float64
	// The amount of weight adjustment
	AddedWeight   float64
}

// Creates new link with specified weight, input and output neurons connected reccurently or not.
func NewLink(weight float64, innode, outnode *NNode, recurrent bool) *Link {
	link := newLink(weight)
	link.InNode = innode
	link.OutNode = outnode
	link.IsRecurrent = recurrent
	return link
}

// Creates new Link with specified Trait
func NewLinkWithTrait(trait *neat.Trait, weight float64, innode, outnode *NNode, recurrent bool) *Link {
	link := newLink(weight)
	link.InNode = innode
	link.OutNode = outnode
	link.IsRecurrent = recurrent
	link.Trait = trait
	link.deriveTrait(trait)
	return link
}

// The copy constructor to create new link with parameters taken from provided ones and connecting specified nodes
func NewLinkCopy(l *Link, innode, outnode *NNode) *Link {
	link := newLink(l.Weight)
	link.InNode = innode
	link.OutNode = outnode
	link.Trait = l.Trait
	link.deriveTrait(l.Trait)
	link.IsRecurrent = l.IsRecurrent
	return link
}

// The private default constructor
func newLink(weight float64) *Link {
	return &Link{
		Weight:weight,
	}
}

// Checks if this link is genetically equal to provided one, i.e. connects nodes with the same IDs and has equal
// recurrent flag. I.e. if both links represent the same Gene.
func (l *Link) IsEqualGenetically(ol *Link) bool {
	same_in_node := (l.InNode.Id == ol.InNode.Id)
	same_out_node := (l.OutNode.Id == ol.OutNode.Id)
	same_recurrent := (l.IsRecurrent == ol.IsRecurrent)

	return same_in_node && same_out_node && same_recurrent
}

// The Link methods implementation
func (l *Link) String() string {
	return fmt.Sprintf("[Link: (%s <-> %s), weight: %.3f, recurrent: %t, time delayed: %t]",
		l.InNode, l.OutNode, l.Weight, l.IsRecurrent, l.IsTimeDelayed)
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