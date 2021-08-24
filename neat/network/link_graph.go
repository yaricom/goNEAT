package network

import (
	"fmt"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
)

// The Gonum Graph specific

// From returns the from node of the edge. Implements graph.Edge From method
func (l *Link) From() graph.Node {
	return l.InNode
}

// To returns the to node of the edge. Implements graph.To From method
func (l *Link) To() graph.Node {
	return l.OutNode
}

// Weight returns weight of this link
func (l *Link) Weight() float64 {
	return l.ConnectionWeight
}

// ReversedEdge returns the edge reversal of the receiver
// if a reversal is valid for the data type.
// When a reversal is valid an edge of the same type as
// the receiver with nodes of the receiver swapped should
// be returned, otherwise the receiver should be returned
// unaltered.
func (l *Link) ReversedEdge() graph.Edge {
	// the reversal is not valid - returning the same
	return l
}

// Attributes returns list of standard attributes associated with the graph edge
func (l *Link) Attributes() []encoding.Attribute {
	attrs := []encoding.Attribute{{
		Key:   "weight",
		Value: fmt.Sprintf("%f", l.ConnectionWeight),
	},
		{
			Key:   "recurrent",
			Value: fmt.Sprintf("%v", l.IsRecurrent),
		},
	}

	if len(l.Params) > 0 {
		attrs = append(attrs, encoding.Attribute{
			Key:   "parameters",
			Value: fmt.Sprintf("%v", l.Params),
		})
	}

	return attrs
}
