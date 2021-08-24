package network

import (
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"gonum.org/v1/gonum/graph/encoding"
)

// The Gonum Graph specific

// ID is to get ID of the node. Implements graph.Node ID method.
func (n *NNode) ID() int64 {
	return int64(n.Id)
}

// Attributes returns list of standard attributes associated with the graph node
func (n *NNode) Attributes() []encoding.Attribute {
	attrs := []encoding.Attribute{{
		Key:   "neuron_type",
		Value: NeuronTypeName(n.NeuronType),
	}}

	if activationFunc, err := math.NodeActivators.ActivationNameFromType(n.ActivationType); err == nil {
		attrs = append(attrs, encoding.Attribute{
			Key:   "activation_type",
			Value: activationFunc,
		})
	}
	if len(n.Params) > 0 {
		attrs = append(attrs, encoding.Attribute{
			Key:   "parameters",
			Value: fmt.Sprintf("%v", n.Params),
		})
	}

	return attrs
}
