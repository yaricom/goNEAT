package network

import (
	"bytes"
	"encoding/json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

const jsonFMNStr = `{"id":0,"name":"","input_neuron_count":2,"sensor_neuron_count":3,"output_neuron_count":2,"bias_neuron_count":1,"total_neuron_count":8,"activation_functions":["SigmoidSteepenedActivation","SigmoidSteepenedActivation","SigmoidSteepenedActivation","SigmoidSteepenedActivation","SigmoidSteepenedActivation","SigmoidSteepenedActivation","SigmoidSteepenedActivation","SigmoidSteepenedActivation"],"bias_list":[0,0,0,0,0,0,1,0],"connections":[{"source_index":1,"target_index":5,"weight":15,"signal":0},{"source_index":2,"target_index":5,"weight":10,"signal":0},{"source_index":2,"target_index":6,"weight":5,"signal":0},{"source_index":6,"target_index":7,"weight":17,"signal":0},{"source_index":5,"target_index":3,"weight":7,"signal":0},{"source_index":7,"target_index":3,"weight":4.5,"signal":0},{"source_index":7,"target_index":4,"weight":13,"signal":0}],"modules":[]}`
const jsonFNMStrModule = `{"id":0,"name":"","input_neuron_count":2,"sensor_neuron_count":3,"output_neuron_count":2,"bias_neuron_count":1,"total_neuron_count":8,"activation_functions":["SigmoidSteepenedActivation","SigmoidSteepenedActivation","SigmoidSteepenedActivation","LinearActivation","LinearActivation","LinearActivation","LinearActivation","NullActivation"],"bias_list":[0,0,0,0,0,10,1,0],"connections":[{"source_index":1,"target_index":5,"weight":15,"signal":0},{"source_index":2,"target_index":6,"weight":5,"signal":0},{"source_index":7,"target_index":3,"weight":4.5,"signal":0},{"source_index":7,"target_index":4,"weight":13,"signal":0}],"modules":[{"activation_type":"MultiplyModuleActivation","input_indexes":[5,6],"output_indexes":[7]}]}`

func TestFastModularNetworkSolver_WriteModel_NoModule(t *testing.T) {
	net := buildNetwork()

	fmm, err := net.FastNetworkSolver()
	require.NoError(t, err, "failed to create fast network solver")

	outBuf := bytes.NewBufferString("")
	err = fmm.(*FastModularNetworkSolver).WriteModel(outBuf)
	require.NoError(t, err, "failed to write model")

	println(outBuf.String())

	var expected interface{}
	err = json.Unmarshal([]byte(jsonFMNStr), &expected)
	require.NoError(t, err, "failed to unmarshal expected json")
	var actual interface{}
	err = json.Unmarshal(outBuf.Bytes(), &actual)
	require.NoError(t, err, "failed to unmarshal actual json")

	assert.EqualValues(t, expected, actual, "model JSON does not match expected JSON")
}

func TestFastModularNetworkSolver_WriteModel_WithModule(t *testing.T) {
	net := buildModularNetwork()

	fmm, err := net.FastNetworkSolver()
	require.NoError(t, err, "failed to create fast network solver")

	outBuf := bytes.NewBufferString("")
	err = fmm.(*FastModularNetworkSolver).WriteModel(outBuf)
	require.NoError(t, err, "failed to write model")

	println(outBuf.String())

	var expected interface{}
	err = json.Unmarshal([]byte(jsonFNMStrModule), &expected)
	require.NoError(t, err, "failed to unmarshal expected json")
	var actual interface{}
	err = json.Unmarshal(outBuf.Bytes(), &actual)
	require.NoError(t, err, "failed to unmarshal actual json")

	assert.EqualValues(t, expected, actual, "model JSON does not match expected JSON")
}
