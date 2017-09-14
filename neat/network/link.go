package network

import (
	"fmt"
)

// A LINK is a connection from one node to another with an associated weight.
// It can be marked as recurrent.
type Link struct {
	// Weight of connection
	Weight float64
	// NNode inputting into the link
	InNode *NNode
	// NNode that the link affects
	OutNode *NNode
	// If TRUE the link is recurrent
	IsRecurrent bool
	// If TRUE the link is time delayed
	IsTimeDelayed bool

	// Points to a trait of parameters for genetic creation
	LinkTrait *Trait

	/* ************ LEARNING PARAMETERS *********** */
	/* These are link-related parameters that change during Hebbian type learning */
	// The amount of weight adjustment
	AddedWeight float64

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
func NewLinkWithTrait(trait *Trait, weight float64, innode, outnode *NNode, recurrent bool) *Link {
	link := newLink(weight)
	link.InNode = innode
	link.OutNode = outnode
	link.IsRecurrent = recurrent
	link.LinkTrait = trait
	return link
}

func NewLinkWeight(weight float64) *Link {
	return newLink(weight)
}

// The private default constructor
func newLink(weight float64) *Link {
	return &Link{
		Weight:weight,
	}
}

// The Link methods implementation
func (l *Link) String() string {
	return fmt.Sprintf("[Link: (%s <-> %s), weight: %.3f, recurrent: %t, time delayed: %t]",
		l.InNode, l.OutNode, l.Weight, l.IsRecurrent, l.IsTimeDelayed)
}