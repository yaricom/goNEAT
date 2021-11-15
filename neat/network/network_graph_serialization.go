package network

import (
	"encoding/json"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/formats/cytoscapejs"
	"io"
)

// WriteDOT is to write this network graph using the GraphViz DOT encoding.
// See DOT Guide: https://www.graphviz.org/pdf/dotguide.pdf
func (n *Network) WriteDOT(w io.Writer) error {
	data, err := dot.Marshal(n, n.Name, "", "")
	if err != nil {
		return err
	}
	if _, err = w.Write(data); err != nil {
		return err
	}
	return nil
}

// WriteCytoscapeJSON is to write this network graph using Cytoscape JSON encoding. Generated JSON file can be used
// for visualization with Cytoscape application https://cytoscape.org or as input to the Cytoscape JavaScript library
// https://js.cytoscape.org
func (n *Network) WriteCytoscapeJSON(w io.Writer) error {
	elements := cytoscapejs.Elements{
		Nodes: make([]cytoscapejs.Node, 0),
		Edges: make([]cytoscapejs.Edge, 0),
	}

	// add all ordinary nodes
	for _, node := range n.allNodes {
		// populate Nodes data
		elements.Nodes = append(elements.Nodes, nodeToCyJsNode(node, false))
		// populate edges data from incoming side
		for _, e := range node.Incoming {
			elements.Edges = append(elements.Edges, linkToCyJsEdge(e))
		}
	}

	// add all control nodes
	for _, node := range n.controlNodes {
		// populate Nodes data
		elements.Nodes = append(elements.Nodes, nodeToCyJsNode(node, true))

		// populate edges data from the incoming side
		for _, e := range node.Incoming {
			elements.Edges = append(elements.Edges, linkToCyJsEdge(e))
		}
		// populate edges data to the outgoing side
		for _, e := range node.Outgoing {
			elements.Edges = append(elements.Edges, linkToCyJsEdge(e))
		}
	}

	// create Cytoscape graph
	graphNodeEdge := cytoscapejs.GraphNodeEdge{
		Elements: elements,
	}
	// marshal to Cytoscape JSON
	if data, err := json.Marshal(graphNodeEdge); err != nil {
		return err
	} else if _, err = w.Write(data); err != nil {
		return err
	}
	return nil
}

func nodeToCyJsNode(node *NNode, control bool) cytoscapejs.Node {
	actName, err := math.NodeActivators.ActivationNameFromType(node.ActivationType)
	if err != nil {
		actName = "unknown"
	}
	return cytoscapejs.Node{
		Data: cytoscapejs.NodeData{
			ID: fmt.Sprintf("%d", node.Id),
			Attributes: map[string]interface{}{
				"activation_value":      node.Activation,
				"activation_function":   actName,
				"neuron_type":           NeuronTypeName(node.NeuronType),
				"node_type":             NodeTypeName(node.NodeType()),
				"in_connections_count":  len(node.Incoming),
				"out_connections_count": len(node.Outgoing),
				"trait":                 node.Trait.String(),
				"control_node":          control,
			},
		},
		Selectable: true,
	}
}

func linkToCyJsEdge(link *Link) cytoscapejs.Edge {
	return cytoscapejs.Edge{
		Data: cytoscapejs.EdgeData{
			ID:     link.IDString(),
			Source: fmt.Sprintf("%d", link.InNode.Id),
			Target: fmt.Sprintf("%d", link.OutNode.Id),
			Attributes: map[string]interface{}{
				"weight":       link.ConnectionWeight,
				"recurrent":    link.IsRecurrent,
				"time_delayed": link.IsTimeDelayed,
				"trait":        link.Trait.String(),
			},
		},
		Selectable: true,
	}
}
