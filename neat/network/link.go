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

// Copy trait parameters into this link's parameters
func (l *Link) deriveTrait(t *neat.Trait) {
	l.Params = make([]float64, neat.Num_trait_params)
	if t != nil {
		for i, p := range t.Params {
			l.Params[i] = p
		}
	}
}