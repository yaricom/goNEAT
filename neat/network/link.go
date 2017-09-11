package network

import (
	"github.com/yaricom/goNEAT/neat"
	"fmt"
)

// A LINK is a connection from one node to another with an associated weight.
// It can be marked as recurrent.
type Link interface {

	// Set added weight
	SetAddedWeight(weight float64)
	// Returns IN node
	InNode() NNode
}

// Creates new link with specified weight, input and output neurons connected reccurently or not.
func NewLink(weight float64, innode, outnode *NNode, recurrent bool) Link {
	link := newLink(weight)
	link.in_node = innode
	link.out_node = outnode
	link.is_recurrent = recurrent
	return link
}

// Creates new Link with specified Trait
func NewLinkWithTrait(trait *Trait, weight float64, innode, outnode *NNode, recurrent bool) Link {
	link := newLink(weight)
	link.in_node = innode
	link.out_node = outnode
	link.is_recurrent = recurrent
	link.linktrait = trait
	return link
}

func NewLinkWeight(weight float64) Link {
	return newLink(weight)
}

// The internal representation
type link struct {
	// Weight of connection
	weight float64
	// NNode inputting into the link
	in_node *NNode
	// NNode that the link affects
	out_node *NNode
	// If TRUE the link is recurrent
	is_recurrent bool
	// If TRUE the link is time delayed
	time_delay bool

	// Points to a trait of parameters for genetic creation
	linktrait *Trait

	/* ************ LEARNING PARAMETERS *********** */
	/* These are link-related parameters that change
	   during Hebbian type learning */
	// The amount of weight adjustment
	added_weight float64
	// The parameters to be learned
	params []float64
}

// The private default constructor
func newLink(weight float64) link {
	return link{
		weight:weight,
		params:make([]float64, neat.Num_trait_params),
	}
}

func (n link) String() string {
	return fmt.Sprintf("(link: %s <-> %s, w: %f", n.in_node, n.out_node, n.weight)
}