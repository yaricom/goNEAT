package formats

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"testing"
)

const jsonDefaultStyle = `{"elements":{"nodes":[{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#339FDC","border-color":"#CCCCCC","control_node":false,"id":"1","in_connections_count":0,"neuron_type":"INPT","node_type":"SENSOR","out_connections_count":1,"parent":"","shape":"diamond"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#339FDC","border-color":"#CCCCCC","control_node":false,"id":"2","in_connections_count":0,"neuron_type":"INPT","node_type":"SENSOR","out_connections_count":2,"parent":"","shape":"diamond"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#FFCC33","border-color":"#CCCCCC","control_node":false,"id":"3","in_connections_count":0,"neuron_type":"BIAS","node_type":"SENSOR","out_connections_count":1,"parent":"","shape":"pentagon"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#009999","border-color":"#CCCCCC","control_node":false,"id":"4","in_connections_count":2,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":1,"parent":"","shape":"hexagon"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#009999","border-color":"#CCCCCC","control_node":false,"id":"5","in_connections_count":2,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":1,"parent":"","shape":"hexagon"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#009999","border-color":"#CCCCCC","control_node":false,"id":"6","in_connections_count":1,"neuron_type":"HIDN","node_type":"NEURON","out_connections_count":2,"parent":"","shape":"hexagon"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#E7298A","border-color":"#CCCCCC","control_node":false,"id":"7","in_connections_count":2,"neuron_type":"OUTP","node_type":"NEURON","out_connections_count":0,"parent":"","shape":"round-rectangle"},"selectable":true},{"data":{"activation_function":"SigmoidSteepenedActivation","activation_value":0,"background-color":"#E7298A","border-color":"#CCCCCC","control_node":false,"id":"8","in_connections_count":1,"neuron_type":"OUTP","node_type":"NEURON","out_connections_count":0,"parent":"","shape":"round-rectangle"},"selectable":true}],"edges":[{"data":{"id":"1-4","recurrent":false,"source":"1","target":"4","time_delayed":false,"weight":15},"selectable":true},{"data":{"id":"2-4","recurrent":false,"source":"2","target":"4","time_delayed":false,"weight":10},"selectable":true},{"data":{"id":"2-5","recurrent":false,"source":"2","target":"5","time_delayed":false,"weight":5},"selectable":true},{"data":{"id":"3-5","recurrent":false,"source":"3","target":"5","time_delayed":false,"weight":1},"selectable":true},{"data":{"id":"5-6","recurrent":false,"source":"5","target":"6","time_delayed":false,"weight":17},"selectable":true},{"data":{"id":"4-7","recurrent":false,"source":"4","target":"7","time_delayed":false,"weight":7},"selectable":true},{"data":{"id":"6-7","recurrent":false,"source":"6","target":"7","time_delayed":false,"weight":4.5},"selectable":true},{"data":{"id":"6-8","recurrent":false,"source":"6","target":"8","time_delayed":false,"weight":13},"selectable":true}]},"layout":{"name":"circle"},"style":[{"selector":"node","style":{"background-color":"data(background-color)","border-color":"data(border-color)","border-width":3,"label":"data(id)","shape":"data(shape)"}},{"selector":"edge","style":{"curve-style":"bezier","line-color":"#CCCCCC","target-arrow-color":"#CCCCCC","target-arrow-shape":"triangle-backcurve","width":5}}]}`

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
