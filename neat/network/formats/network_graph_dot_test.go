package formats

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"testing"
)

func buildNetwork() *network.Network {
	allNodes := []*network.NNode{
		network.NewNNode(1, network.InputNeuron),
		network.NewNNode(2, network.InputNeuron),
		network.NewNNode(3, network.BiasNeuron),
		network.NewNNode(4, network.HiddenNeuron),
		network.NewNNode(5, network.HiddenNeuron),
		network.NewNNode(6, network.HiddenNeuron),
		network.NewNNode(7, network.OutputNeuron),
		network.NewNNode(8, network.OutputNeuron),
	}

	// HIDDEN 4
	allNodes[3].ConnectFrom(allNodes[0], 15.0)
	allNodes[3].ConnectFrom(allNodes[1], 10.0)
	// HIDDEN 5
	allNodes[4].ConnectFrom(allNodes[1], 5.0)
	allNodes[4].ConnectFrom(allNodes[2], 1.0)
	// HIDDEN 6
	allNodes[5].ConnectFrom(allNodes[4], 17.0)
	// OUTPUT 7
	allNodes[6].ConnectFrom(allNodes[3], 7.0)
	allNodes[6].ConnectFrom(allNodes[5], 4.5)
	// OUTPUT 8
	allNodes[7].ConnectFrom(allNodes[5], 13.0)

	return network.NewNetwork(allNodes[0:3], allNodes[6:8], allNodes, 0)
}

func TestNetwork_WriteDOT(t *testing.T) {
	net := buildNetwork()
	net.Name = "TestNN"

	b := bytes.NewBufferString("")
	err := WriteDOT(b, net)
	require.NoError(t, err, "failed to DOT encode")
	t.Log(b)
	assert.NotEmpty(t, b)
}
