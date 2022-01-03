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

const (
	attrActivationValue        = "activation_value"
	attrActivationFunc         = "activation_function"
	attrNeuronType             = "neuron_type"
	attrNodeType               = "node_type"
	attrInputConnectionsCount  = "in_connections_count"
	attrOutputConnectionsCount = "out_connections_count"
	attrControlNode            = "control_node"
	attrBackgroundColor        = "background-color"
	attrBorderColor            = "border-color"
	attrShape                  = "shape"
	attrTrait                  = "trait"
)

func nodeToCyJsNode(node *network.NNode, control bool) cytoscapejs.Node {
	actName, err := math.NodeActivators.ActivationNameFromType(node.ActivationType)
	if err != nil {
		actName = "unknown"
	}
	nodeJS := cytoscapejs.Node{
		Data: cytoscapejs.NodeData{
			ID: fmt.Sprintf("%d", node.Id),
			Attributes: map[string]interface{}{
				attrActivationValue:        node.Activation,
				attrActivationFunc:         actName,
				attrNeuronType:             network.NeuronTypeName(node.NeuronType),
				attrNodeType:               network.NodeTypeName(node.NodeType()),
				attrInputConnectionsCount:  len(node.Incoming),
				attrOutputConnectionsCount: len(node.Outgoing),
				attrControlNode:            control,
				attrBackgroundColor:        nodeBgColor(node, control),
				attrBorderColor:            nodeBorderColor(node, control),
				attrShape:                  nodeShape(node, control),
			},
		},
		Selectable: true,
	}
	if node.Trait != nil {
		nodeJS.Data.Attributes[attrTrait] = node.Trait.String()
	}
	return nodeJS
}

func linkToCyJsEdge(link *network.Link) cytoscapejs.Edge {
	edgeJS := cytoscapejs.Edge{
		Data: cytoscapejs.EdgeData{
			ID:     link.IDString(),
			Source: fmt.Sprintf("%d", link.InNode.Id),
			Target: fmt.Sprintf("%d", link.OutNode.Id),
			Attributes: map[string]interface{}{
				"weight":       link.ConnectionWeight,
				"recurrent":    link.IsRecurrent,
				"time_delayed": link.IsTimeDelayed,
			},
		},
		Selectable: true,
	}
	if link.Trait != nil {
		edgeJS.Data.Attributes["trait"] = link.Trait.String()
	}
	return edgeJS
}

const (
	colorControl = "#EA1E53"
	colorInput   = "#339FDC"
	colorOutput  = "#E7298A"
	colorHidden  = "#009999"
	colorBias    = "#FFCC33"
	colorDefault = "#555"
)

func nodeBgColor(node *network.NNode, control bool) string {
	if control {
		return colorControl
	}
	switch node.NeuronType {
	case network.InputNeuron:
		return colorInput
	case network.OutputNeuron:
		return colorOutput
	case network.HiddenNeuron:
		return colorHidden
	case network.BiasNeuron:
		return colorBias
	}
	return colorDefault
}

const (
	shapeControl = "octagon"
	shapeInput   = "diamond"
	shapeOutput  = "round-rectangle"
	shapeHidden  = "hexagon"
	shapeBias    = "pentagon"
	shapeDefault = "ellipse"
)

func nodeShape(node *network.NNode, control bool) string {
	if control {
		return shapeControl
	}
	switch node.NeuronType {
	case network.InputNeuron:
		return shapeInput
	case network.OutputNeuron:
		return shapeOutput
	case network.HiddenNeuron:
		return shapeHidden
	case network.BiasNeuron:
		return shapeBias
	}
	return shapeDefault
}

const (
	borderColorControl = "#AAAAAA"
	borderColorOther   = "#CCCCCC"
)

func nodeBorderColor(_ *network.NNode, control bool) string {
	if control {
		return borderColorControl
	} else {
		return borderColorOther
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
