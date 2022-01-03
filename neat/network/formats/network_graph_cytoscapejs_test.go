package formats

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"testing"
)

const (
	jsonDefaultStyle = `{"elements":{"nodes":[{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#339FDC","border-color":"#CCCCCC","control_node":false,"id":"1","in_connections_count":0,"neuron_type":"INPT","node_type":"SENSOR","out_connections_count":1,"parent":"","shape":"diamond"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#339FDC","border-color":"#CCCCCC","control_node":false,"id":"2","in_connections_count":0,"neuron_type":"INPT","node_type":"SENSOR","out_connections_count":2,"parent":"","shape":"diamond"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#FFCC33","border-color":"#CCCCCC","control_node":false,"id":"3","in_connections_count":0,"neuron_type":"BIAS","node_type":"SENSOR","out_connections_count":1,"parent":"","shape":"pentagon"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#009999","border-color":"#CCCCCC","control_node":false,"id":"4","in_connections_count":2,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":1,"parent":"","shape":"hexagon"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#009999","border-color":"#CCCCCC","control_node":false,"id":"5","in_connections_count":2,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":1,"parent":"","shape":"hexagon"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#009999","border-color":"#CCCCCC","control_node":false,"id":"6","in_connections_count":1,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":2,"parent":"","shape":"hexagon"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#E7298A","border-color":"#CCCCCC","control_node":false,"id":"7","in_connections_count":2,"neuron_type":"OUTP","node_type":"NEURON","out_connections_count":0,"parent":"","shape":"round-rectangle"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#E7298A","border-color":"#CCCCCC","control_node":false,"id":"8","in_connections_count":1,"neuron_type":"OUTP","node_type":"NEURON","out_connections_count":0,"parent":"","shape":"round-rectangle"},"selectable":true}],"edges":[{"data":{"id":"1-4","recurrent":false,"source":"1","target":"4","time_delayed":false,"weight":15},"selectable":true},{"data":{"id":"2-4","recurrent":false,"source":"2","target":"4","time_delayed":false,"weight":10},"selectable":true},{"data":{"id":"2-5","recurrent":false,"source":"2","target":"5","time_delayed":false,"weight":5},"selectable":true},{"data":{"id":"3-5","recurrent":false,"source":"3","target":"5","time_delayed":false,"weight":1},"selectable":true},{"data":{"id":"5-6","recurrent":false,"source":"5","target":"6","time_delayed":false,"weight":17},"selectable":true},{"data":{"id":"4-7","recurrent":false,"source":"4","target":"7","time_delayed":false,"weight":7},"selectable":true},{"data":{"id":"6-7","recurrent":false,"source":"6","target":"7","time_delayed":false,"weight":4.5},"selectable":true},{"data":{"id":"6-8","recurrent":false,"source":"6","target":"8","time_delayed":false,"weight":13},"selectable":true}]},"layout":{"name":"circle"},"style":[{"selector":"node","style":{"background-color":"data(background-color)","border-color":"data(border-color)","border-width":3,"label":"data(id)","shape":"data(shape)"}},{"selector":"edge","style":{"curve-style":"bezier","line-color":"#CCCCCC","target-arrow-color":"#CCCCCC","target-arrow-shape":"triangle-backcurve","width":5}}]}`
	jsonModular      = `{"elements":{"nodes":[{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#339FDC","border-color":"#CCCCCC","control_node":false,"id":"1","in_connections_count":0,"neuron_type":"INPT","node_type":"SENSOR","out_connections_count":1,"parent":"","shape":"diamond"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#339FDC","border-color":"#CCCCCC","control_node":false,"id":"2","in_connections_count":0,"neuron_type":"INPT","node_type":"SENSOR","out_connections_count":1,"parent":"","shape":"diamond"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#FFCC33","border-color":"#CCCCCC","control_node":false,"id":"3","in_connections_count":0,"neuron_type":"BIAS","node_type":"SENSOR","out_connections_count":2,"parent":"","shape":"pentagon"},"selectable":true},{"data":{"activation_function":"LinearActivation","activation_value":0,"background-color":"#009999","border-color":"#CCCCCC","control_node":false,"id":"4","in_connections_count":2,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":0,"parent":"","shape":"hexagon"},"selectable":true},{"data":{"activation_function":"LinearActivation","activation_value":0,"background-color":"#009999","border-color":"#CCCCCC","control_node":false,"id":"5","in_connections_count":2,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":0,"parent":"","shape":"hexagon"},"selectable":true},{"data":{"activation_function":"NullActivation","activation_value":0,"background-color":"#009999","border-color":"#CCCCCC","control_node":false,"id":"7","in_connections_count":0,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":2,"parent":"","shape":"hexagon"},"selectable":true},{"data":{"activation_function":"LinearActivation","activation_value":0,"background-color":"#E7298A","border-color":"#CCCCCC","control_node":false,"id":"8","in_connections_count":1,"neuron_type":"OUTP","node_type":"NEURON","out_connections_count":0,"parent":"","shape":"round-rectangle"},"selectable":true},{"data":{"activation_function":"LinearActivation","activation_value":0,"background-color":"#E7298A","border-color":"#CCCCCC","control_node":false,"id":"9","in_connections_count":1,"neuron_type":"OUTP","node_type":"NEURON","out_connections_count":0,"parent":"","shape":"round-rectangle"},"selectable":true},{"data":{"activation_function":"MultiplyModuleActivation","activation_value":0,"background-color":"#EA1E53","border-color":"#AAAAAA","control_node":true,"id":"6","in_connections_count":2,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":1,"parent":"","shape":"octagon"},"selectable":true}],"edges":[{"data":{"id":"1-4","recurrent":false,"source":"1","target":"4","time_delayed":false,"weight":15},"selectable":true},{"data":{"id":"3-4","recurrent":false,"source":"3","target":"4","time_delayed":false,"weight":10},"selectable":true},{"data":{"id":"2-5","recurrent":false,"source":"2","target":"5","time_delayed":false,"weight":5},"selectable":true},{"data":{"id":"3-5","recurrent":false,"source":"3","target":"5","time_delayed":false,"weight":1},"selectable":true},{"data":{"id":"7-8","recurrent":false,"source":"7","target":"8","time_delayed":false,"weight":4.5},"selectable":true},{"data":{"id":"7-9","recurrent":false,"source":"7","target":"9","time_delayed":false,"weight":13},"selectable":true},{"data":{"id":"4-6","recurrent":false,"source":"4","target":"6","time_delayed":false,"weight":1},"selectable":true},{"data":{"id":"5-6","recurrent":false,"source":"5","target":"6","time_delayed":false,"weight":1},"selectable":true},{"data":{"id":"6-7","recurrent":false,"source":"6","target":"7","time_delayed":false,"weight":1},"selectable":true}]},"layout":{"name":"circle"},"style":[{"selector":"node","style":{"background-color":"data(background-color)","border-color":"data(border-color)","border-width":3,"label":"data(id)","shape":"data(shape)"}},{"selector":"edge","style":{"curve-style":"bezier","line-color":"#CCCCCC","target-arrow-color":"#CCCCCC","target-arrow-shape":"triangle-backcurve","width":5}}]}`
)

func buildModularNetwork() *network.Network {
	allNodes := []*network.NNode{
		network.NewNNode(1, network.InputNeuron),
		network.NewNNode(2, network.InputNeuron),
		network.NewNNode(3, network.BiasNeuron),
		network.NewNNode(4, network.HiddenNeuron),
		network.NewNNode(5, network.HiddenNeuron),
		network.NewNNode(7, network.HiddenNeuron),
		network.NewNNode(8, network.OutputNeuron),
		network.NewNNode(9, network.OutputNeuron),
	}
	controlNodes := []*network.NNode{
		network.NewNNode(6, network.HiddenNeuron),
	}
	// HIDDEN 6 - control node
	controlNodes[0].ActivationType = math.MultiplyModuleActivation
	controlNodes[0].AddIncoming(allNodes[3], 1.0)
	controlNodes[0].AddIncoming(allNodes[4], 1.0)
	controlNodes[0].AddOutgoing(allNodes[5], 1.0)

	// HIDDEN 4
	allNodes[3].ActivationType = math.LinearActivation
	allNodes[3].ConnectFrom(allNodes[0], 15.0)
	allNodes[3].ConnectFrom(allNodes[2], 10.0)
	// HIDDEN 5
	allNodes[4].ActivationType = math.LinearActivation
	allNodes[4].ConnectFrom(allNodes[1], 5.0)
	allNodes[4].ConnectFrom(allNodes[2], 1.0)

	// HIDDEN 7
	allNodes[5].ActivationType = math.NullActivation

	// OUTPUT 8
	allNodes[6].ConnectFrom(allNodes[5], 4.5)
	allNodes[6].ActivationType = math.LinearActivation
	// OUTPUT 9
	allNodes[7].ConnectFrom(allNodes[5], 13.0)
	allNodes[7].ActivationType = math.LinearActivation

	return network.NewModularNetwork(allNodes[0:3], allNodes[6:8], allNodes, controlNodes, 0)
}

func TestWriteCytoscapeJSON(t *testing.T) {
	net := buildNetwork()

	b := bytes.NewBufferString("")
	err := WriteCytoscapeJSON(b, net)
	assert.NoError(t, err)
	assert.NotEmpty(t, b)
	assert.Equal(t, jsonDefaultStyle, b.String())
}

func TestWriteCytoscapeJSONWithStyle(t *testing.T) {
	net := buildNetwork()

	style := &CytoscapeStyleOptions{
		Style:  []ElementStyle{defaultNodeStyle(), defaultEdgeStyle()},
		Layout: defaultLayout(),
	}
	b := bytes.NewBufferString("")
	err := WriteCytoscapeJSONWithStyle(b, net, style)
	assert.NoError(t, err)
	assert.NotEmpty(t, b)
	assert.Equal(t, jsonDefaultStyle, b.String())
}

func TestWriteCytoscapeJSON_Modular(t *testing.T) {
	net := buildModularNetwork()

	b := bytes.NewBufferString("")
	err := WriteCytoscapeJSON(b, net)
	assert.NoError(t, err)
	assert.NotEmpty(t, b)
	assert.Equal(t, jsonModular, b.String())
}

func TestWriteCytoscapeJSON_nodeShape(t *testing.T) {
	node := network.NewNNode(1, network.InputNeuron)

	testCases := []struct {
		nodeType network.NodeNeuronType
		control  bool
		shape    string
	}{
		{
			nodeType: network.InputNeuron,
			control:  true,
			shape:    shapeControl,
		},
		{
			nodeType: network.InputNeuron,
			control:  false,
			shape:    shapeInput,
		},
		{
			nodeType: network.HiddenNeuron,
			control:  false,
			shape:    shapeHidden,
		},
		{
			nodeType: network.BiasNeuron,
			control:  false,
			shape:    shapeBias,
		},
		{
			nodeType: network.OutputNeuron,
			control:  false,
			shape:    shapeOutput,
		},
		{
			nodeType: network.BiasNeuron + 1,
			control:  false,
			shape:    shapeDefault,
		},
	}

	for i, tc := range testCases {
		node.NeuronType = tc.nodeType
		shape := nodeShape(node, tc.control)
		assert.Equal(t, tc.shape, shape, "wrong shape at: %d", i)
	}
}

func TestWriteCytoscapeJSON_nodeBgColor(t *testing.T) {
	node := network.NewNNode(1, network.InputNeuron)

	testCases := []struct {
		nodeType network.NodeNeuronType
		control  bool
		color    string
	}{
		{
			nodeType: network.InputNeuron,
			control:  true,
			color:    colorControl,
		},
		{
			nodeType: network.InputNeuron,
			control:  false,
			color:    colorInput,
		},
		{
			nodeType: network.HiddenNeuron,
			control:  false,
			color:    colorHidden,
		},
		{
			nodeType: network.BiasNeuron,
			control:  false,
			color:    colorBias,
		},
		{
			nodeType: network.OutputNeuron,
			control:  false,
			color:    colorOutput,
		},
		{
			nodeType: network.BiasNeuron + 1,
			control:  false,
			color:    colorDefault,
		},
	}
	for i, tc := range testCases {
		node.NeuronType = tc.nodeType
		color := nodeBgColor(node, tc.control)
		assert.Equal(t, tc.color, color, "wrong node background color at: %d", i)
	}
}

func TestWriteCytoscapeJSON_nodeBorderColor(t *testing.T) {
	node := network.NewNNode(1, network.InputNeuron)

	color := nodeBorderColor(node, true)
	assert.Equal(t, borderColorControl, color)

	color = nodeBorderColor(node, false)
	assert.Equal(t, borderColorOther, color)
}

func TestWriteCytoscapeJSON_nodeToCyJsNode(t *testing.T) {
	node := network.NewNNode(1, network.InputNeuron)
	node.Trait = neat.NewTrait()

	testCases := []struct {
		control bool
	}{
		{
			control: false,
		},
		{
			control: true,
		},
	}

	for i, tc := range testCases {
		t.Logf("Test case: %d", i)

		node.ActivationType = math.SigmoidApproximationActivation
		nodeJS := nodeToCyJsNode(node, tc.control)
		require.NotNil(t, nodeJS)
		require.NotEmpty(t, nodeJS.Data.Attributes)

		actName, err := math.NodeActivators.ActivationNameFromType(node.ActivationType)
		require.NoError(t, err)
		attrs := map[string]interface{}{
			attrActivationValue:        node.Activation,
			attrActivationFunc:         actName,
			attrNeuronType:             network.NeuronTypeName(node.NeuronType),
			attrNodeType:               network.NodeTypeName(node.NodeType()),
			attrInputConnectionsCount:  len(node.Incoming),
			attrOutputConnectionsCount: len(node.Outgoing),
			attrControlNode:            tc.control,
			attrBackgroundColor:        nodeBgColor(node, tc.control),
			attrBorderColor:            nodeBorderColor(node, tc.control),
			attrShape:                  nodeShape(node, tc.control),
			attrTrait:                  node.Trait.String(),
		}
		assert.Equal(t, attrs, nodeJS.Data.Attributes)

		// check unknown activation type
		node.ActivationType = math.MinModuleActivation + 1
		nodeJS = nodeToCyJsNode(node, tc.control)
		require.NotNil(t, nodeJS)
		require.NotEmpty(t, nodeJS.Data.Attributes)

		attrs[attrActivationFunc] = "unknown"
		assert.Equal(t, attrs, nodeJS.Data.Attributes)
	}
}
