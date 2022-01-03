package formats

import (
	"errors"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
)

const alwaysErrorText = "always be failing"

var alwaysError = errors.New(alwaysErrorText)

type ErrorWriter int

func (e ErrorWriter) Write(_ []byte) (int, error) {
	return 0, alwaysError
}

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
