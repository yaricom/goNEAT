package formats

import (
	"encoding/json"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"gonum.org/v1/gonum/graph/formats/cytoscapejs"
	"io"
)

// CytoscapeStyleOptions is to hold style options to be appended to the graph elements definition when serializing to
// Cytoscape JSON. Form more details, see https://js.cytoscape.org/#getting-started/specifying-basic-options
type CytoscapeStyleOptions struct {
	// Style the style to be applied to the graph elements: nodes and edges.
	Style []ElementStyle
	// Layout the layout used for the graph visualization. See https://js.cytoscape.org/#layouts
	Layout interface{}
}

// ElementStyle is to define style of particular element: edge or node.  See https://js.cytoscape.org/#style
type ElementStyle struct {
	//// Selector is to select element to apply style: node or edge
	Selector string `json:"selector"`
	//// Style the map with style options to be applied
	Style map[string]interface{} `json:"style"`
}

// WriteCytoscapeJSON is to write this network graph using Cytoscape JSON encoding. Generated JSON file can be used
// for visualization with Cytoscape application https://cytoscape.org or as input to the Cytoscape JavaScript library
// https://js.cytoscape.org
// This will use goNEAT default style for the graph. If you want to apply different style you can
// use WriteCytoscapeJSONWithStyle and provide your style as a parameter.
func WriteCytoscapeJSON(w io.Writer, n *network.Network) error {
	style := &CytoscapeStyleOptions{
		Style:  []ElementStyle{defaultNodeStyle(), defaultEdgeStyle()},
		Layout: defaultLayout(),
	}
	return WriteCytoscapeJSONWithStyle(w, n, style)
}

// WriteCytoscapeJSONWithStyle allows writing of this network graph using Cytoscape JSON encoding.
// Additionally, it is possible to provide style to be used for rendering of the graph.
// For more details about style, see https://js.cytoscape.org/#getting-started/specifying-basic-options
func WriteCytoscapeJSONWithStyle(w io.Writer, n *network.Network, style *CytoscapeStyleOptions) error {
	elements := cytoscapejs.Elements{
		Nodes: make([]cytoscapejs.Node, 0),
		Edges: make([]cytoscapejs.Edge, 0),
	}

	// add all ordinary nodes
	for _, node := range n.BaseNodes() {
		// populate Nodes data
		elements.Nodes = append(elements.Nodes, nodeToCyJsNode(node, false))
		// populate edges data from incoming side
		for _, e := range node.Incoming {
			elements.Edges = append(elements.Edges, linkToCyJsEdge(e))
		}
	}

	// add all control nodes
	for _, node := range n.ControlNodes() {
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
	if style != nil {
		graphNodeEdge.Layout = style.Layout
		if len(style.Style) > 0 {
			// copy styles if appropriate
			graphNodeEdge.Style = make([]interface{}, 0)
			for _, s := range style.Style {
				graphNodeEdge.Style = append(graphNodeEdge.Style, s)
			}
		}
	}

	// marshal to Cytoscape JSON
	if data, err := json.Marshal(graphNodeEdge); err != nil {
		return err
	} else if _, err = w.Write(data); err != nil {
		return err
	}
	return nil
}

func nodeToCyJsNode(node *network.NNode, control bool) cytoscapejs.Node {
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
				"neuron_type":           network.NeuronTypeName(node.NeuronType),
				"node_type":             network.NodeTypeName(node.NodeType()),
				"in_connections_count":  len(node.Incoming),
				"out_connections_count": len(node.Outgoing),
				"trait":                 node.Trait.String(),
				"control_node":          control,
				"background-color":      nodeBgColor(node, control),
				"border-color":          nodeBorderColor(node, control),
				"shape":                 nodeShape(node, control),
			},
		},
		Selectable: true,
	}
}

func linkToCyJsEdge(link *network.Link) cytoscapejs.Edge {
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

func nodeBgColor(node *network.NNode, control bool) string {
	if control {
		return "#EA1E53"
	}
	switch node.NeuronType {
	case network.InputNeuron:
		return "#339FDC"
	case network.OutputNeuron:
		return "#E7298A"
	case network.HiddenNeuron:
		return "#009999"
	case network.BiasNeuron:
		return "#FFCC33"
	}
	return "#555"
}

func nodeShape(node *network.NNode, control bool) string {
	if control {
		return "octagon"
	}
	switch node.NeuronType {
	case network.InputNeuron:
		return "diamond"
	case network.OutputNeuron:
		return "round-rectangle"
	case network.HiddenNeuron:
		return "hexagon"
	case network.BiasNeuron:
		return "pentagon"
	}
	return "ellipse"
}

func nodeBorderColor(_ *network.NNode, control bool) string {
	if control {
		return "#AAAAAA"
	} else {
		return "#CCCCCC"
	}
}

func defaultNodeStyle() ElementStyle {
	style := map[string]interface{}{
		"shape":            "data(shape)",
		"background-color": "data(background-color)",
		"border-color":     "data(border-color)",
		"border-width":     3.0,
		"label":            "data(id)",
	}
	return ElementStyle{Selector: "node", Style: style}
}

func defaultEdgeStyle() ElementStyle {
	lineColor := "#CCCCCC"
	style := map[string]interface{}{
		"width":              5.0,
		"curve-style":        "bezier",
		"line-color":         lineColor,
		"target-arrow-shape": "triangle-backcurve",
		"target-arrow-color": lineColor,
	}
	return ElementStyle{Selector: "edge", Style: style}
}

func defaultLayout() map[string]interface{} {
	return map[string]interface{}{
		"name": "circle",
	}
}
