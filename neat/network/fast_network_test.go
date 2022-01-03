package network

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestFastModularNetworkSolver_LoadSensors(t *testing.T) {
	net := buildNetwork()

	fmm, err := net.FastNetworkSolver()
	require.NoError(t, err, "failed to create fast network solver")

	// test normal
	data := []float64{0.5, 1.1}
	err = fmm.LoadSensors(data)
	require.NoError(t, err, "failed to load sensors")

	// test abnormal
	data = append(data, 1.0)
	err = fmm.LoadSensors(data)
	require.EqualError(t, err, ErrNetUnsupportedSensorsArraySize.Error())
}

func TestFastModularNetworkSolver_RecursiveSteps(t *testing.T) {
	net := buildNetwork()

	// Create network solver
	data := []float64{0.5, 1.1} // BIAS is 1.0 by definition
	fmm, err := net.FastNetworkSolver()
	require.NoError(t, err, "failed to create fast network solver")
	err = fmm.LoadSensors(data)
	require.NoError(t, err, "failed to load sensors")

	// Activate objective network
	//
	data = append(data, 1.0) // BIAS is a third object
	err = net.LoadSensors(data)
	require.NoError(t, err, "failed to load sensors")

	depth, err := net.MaxActivationDepth()
	require.NoError(t, err, "failed to calculate max depth")

	t.Logf("depth: %d\n", depth)
	logNetworkActivationPath(net, t)

	res, err := net.ForwardSteps(depth)
	require.NoError(t, err, "error when trying to activate objective network")
	require.True(t, res, "failed to activate objective network")

	// Do recursive activation of the Fast Network Solver
	//
	res, err = fmm.RecursiveSteps()
	require.NoError(t, err, "error when trying to activate Fast Network Solver")
	require.True(t, res, "recursive activation failed")

	// Compare activations of objective network and Fast Network Solver
	//
	fmmOutputs := fmm.ReadOutputs()
	require.Equal(t, len(net.Outputs), len(fmmOutputs))

	for i, out := range fmmOutputs {
		assert.Equal(t, net.Outputs[i].Activation, out, "wrong activation at: %d", i)
	}
}

func TestFastModularNetworkSolver_ForwardSteps(t *testing.T) {
	net := buildModularNetwork()

	// create network solver
	data := []float64{1.0, 2.0} // bias inherent
	fmm, err := net.FastNetworkSolver()
	require.NoError(t, err, "failed to create fast network solver")
	err = fmm.LoadSensors(data)
	require.NoError(t, err, "failed to load sensors")

	depth, err := net.MaxActivationDepth()
	require.NoError(t, err, "failed to calculate max depth")

	t.Logf("depth: %d\n", depth)
	logNetworkActivationPath(net, t)

	// activate objective network
	//
	data = append(data, 1.0) // BIAS is third object
	err = net.LoadSensors(data)
	require.NoError(t, err, "failed to load sensors")
	res, err := net.ForwardSteps(depth)
	require.NoError(t, err, "error when trying to activate objective network")
	require.True(t, res, "failed to activate objective network")

	// do forward steps through the solver and test results
	//
	res, err = fmm.ForwardSteps(depth)
	require.NoError(t, err, "error while do forward steps")
	require.True(t, res, "forward steps returned false")

	// check results by comparing activations of objective network and fast network solver
	//
	outputs := fmm.ReadOutputs()
	for i, out := range outputs {
		assert.Equal(t, net.Outputs[i].Activation, out, "wrong activation at: %d", i)
	}
}

func TestFastModularNetworkSolver_Relax(t *testing.T) {
	net := buildModularNetwork()

	// create network solver
	data := []float64{1.5, 2.0} // bias inherent
	fmm, err := net.FastNetworkSolver()
	require.NoError(t, err, "failed to create fast network solver")
	err = fmm.LoadSensors(data)
	require.NoError(t, err, "failed to load sensors")

	depth, err := net.MaxActivationDepth()
	require.NoError(t, err, "failed to calculate max depth")

	t.Logf("depth: %d\n", depth)
	logNetworkActivationPath(net, t)

	// activate objective network
	//
	data = append(data, 1.0) // BIAS is third object
	err = net.LoadSensors(data)
	require.NoError(t, err, "failed to load sensors")
	res, err := net.ForwardSteps(depth)
	require.NoError(t, err, "error when trying to activate objective network")
	require.True(t, res, "failed to activate objective network")

	// do relaxation of fast network solver
	//
	res, err = fmm.Relax(depth, 1)
	require.NoError(t, err)
	require.True(t, res, "failed to relax within given maximal steps number")

	// check results by comparing activations of objective network and fast network solver
	//
	outputs := fmm.ReadOutputs()
	for i, out := range outputs {
		assert.Equal(t, net.Outputs[i].Activation, out, "wrong activation at: %d", i)
	}
}

func TestFastModularNetworkSolver_Flush(t *testing.T) {
	net := buildModularNetwork()

	// create network solver
	data := []float64{1.5, 2.0} // bias inherent
	fmm, err := net.FastNetworkSolver()
	require.NoError(t, err, "failed to create fast network solver")
	err = fmm.LoadSensors(data)
	require.NoError(t, err, "failed to load sensors")

	fmmImpl := fmm.(*FastModularNetworkSolver)
	// test that network has active signals
	active := countActiveSignals(fmmImpl)
	assert.NotZero(t, active, "no active signal found")

	// flush and test
	res, err := fmm.Flush()
	require.NoError(t, err)
	require.True(t, res, "failed to flush network")

	active = countActiveSignals(fmmImpl)
	assert.Zero(t, active, "after flush the active signal still present")
}

func TestFastModularNetworkSolver_NodeCount(t *testing.T) {
	net := buildModularNetwork()

	fmm, err := net.FastNetworkSolver()
	require.NoError(t, err, "failed to create fast network solver")
	assert.Equal(t, 9, fmm.NodeCount())
}

func TestFastModularNetworkSolver_LinkCount(t *testing.T) {
	net := buildModularNetwork()

	fmm, err := net.FastNetworkSolver()
	require.NoError(t, err, "failed to create fast network solver")
	assert.Equal(t, 9, fmm.LinkCount())
}

func countActiveSignals(impl *FastModularNetworkSolver) int {
	active := 0
	for i := impl.biasNeuronCount; i < impl.totalNeuronCount; i++ {
		if impl.neuronSignals[i] != 0.0 {
			active++
		}
	}
	return active
}
