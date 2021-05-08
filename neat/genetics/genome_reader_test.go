package genetics

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"os"
	"strings"
	"testing"
)

func TestPlainGenomeReader_Read(t *testing.T) {
	r, err := NewGenomeReader(strings.NewReader(gnomeStr), PlainGenomeEncoding)
	require.NoError(t, err, "failed to create reader")
	gnome, err := r.Read()
	require.NoError(t, err, "failed to read genome")
	require.NotNil(t, gnome, "genome not initialized")

	// check traits
	//
	require.Len(t, gnome.Traits, 3)
	ids := []int{1, 3, 2}
	for i, tr := range gnome.Traits {
		assert.Equal(t, ids[i], tr.Id, "Wrong Trait ID at: %d", i)
		require.Len(t, tr.Params, 8, "Wrong Trait's parameters length at: %d", i)

		expectedParams := []float64{float64(ids[i]) / 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
		assert.Equal(t, expectedParams, tr.Params, "Wrong Trait params read at: %d", i)
	}

	// check nodes
	//
	require.Len(t, gnome.Nodes, 4)
	for i, n := range gnome.Nodes {
		assert.Equal(t, i+1, n.Id, "Wrong NNode Id at: %d", i)
		if i < 2 {
			// INPUT
			assert.True(t, n.IsSensor(), "Wrong NNode type, SENSOR expected at: %d", i)
			assert.Equal(t, network.InputNeuron, n.NeuronType, "Wrong input Neuron type at: %d", i)
		}
		if i == 2 {
			// BIAS
			assert.True(t, n.IsSensor(), "Wrong NNode type, SENSOR expected at: %d", i)
			assert.Equal(t, network.BiasNeuron, n.NeuronType, "Wrong bias Neuron type at: %d", i)
		}
		if i == 3 {
			// OUTPUT
			assert.True(t, n.IsNeuron(), "Wrong NNode type, NEURON expected at: %d", i)
			assert.Equal(t, network.OutputNeuron, n.NeuronType, "Wrong NNode placement at: %d", i)
		}
	}

	// check genes
	//
	require.Len(t, gnome.Genes, 3)

	for i, g := range gnome.Genes {
		assert.Equal(t, i+1, g.Link.Trait.Id, "Gene Link Trait Id is wrong at: %d", i)
		assert.Equal(t, i+1, g.Link.InNode.Id, "Gene link's input node Id is wrong at: %d", i)
		assert.Equal(t, 4, g.Link.OutNode.Id, "Gene link's output node Id is wrong at: &d", i)
		assert.Equal(t, float64(i)+1.5, g.Link.Weight, "Gene link's weight is wrong at: %d", i)
		assert.False(t, g.Link.IsRecurrent, "Gene link's recurrent flag is wrong at: %d", i)
		assert.EqualValues(t, i+1, g.InnovationNum, "Gene's innovation number is wrong at: %d", i)
		assert.EqualValues(t, 0, g.MutationNum, "Gene's mutation number is wrong at: %d", i)
		assert.True(t, g.IsEnabled, "Gene's enabled flag is wrong at: %d", i)
	}
}

func TestReadGene_readPlainTrait(t *testing.T) {
	params := []float64{
		0.40227575878298616, 0.0, 0.0, 0.0, 0.0, 0.3245553261200018, 0.0, 0.12248956525856575,
	}
	traitId := 2
	traitStr := fmt.Sprintf("%d %g %g %g %g %g %g %g %g",
		traitId, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
	trait, err := readPlainTrait(strings.NewReader(traitStr))
	require.NoError(t, err)
	assert.Equal(t, traitId, trait.Id, "wrong trait ID")
	assert.Equal(t, params, trait.Params, "wrong trait params")
}

// Tests how NNode read working
func TestReadGene_ReadPlainNNode(t *testing.T) {
	nodeId, traitId, nodeType, genNodeLabel := 1, 10, network.SensorNode, network.InputNeuron
	nodeStr := fmt.Sprintf("%d %d %d %d", nodeId, traitId, nodeType, genNodeLabel)

	trait := neat.NewTrait()
	trait.Id = 10
	traits := []*neat.Trait{trait}

	node, err := readPlainNetworkNode(strings.NewReader(nodeStr), traits)
	require.NoError(t, err, "failed to read network node")

	assert.Equal(t, nodeId, node.Id, "wrong node ID")
	assert.Equal(t, trait, node.Trait, "wrong Trait found in the node")
	assert.Equal(t, nodeType, node.NodeType(), "wrong node type")
	assert.Equal(t, genNodeLabel, node.NeuronType, "wrong node placement label (neuron type) found")
}

// Tests Gene ReadGene
func TestReadGene_ReadPlainGene(t *testing.T) {
	// gene  1 1 4 1.1983046913458986 0 1.0 1.1983046913458986 0
	traitId, inNodeId, outNodeId, innovationNum := 1, 1, 4, int64(1)
	weight, mutNum := 1.1983046913458986, 1.1983046913458986
	recurrent, enabled := false, false
	geneStr := fmt.Sprintf("%d %d %d %g %t %d %g %t",
		traitId, inNodeId, outNodeId, weight, recurrent, innovationNum, mutNum, enabled)

	trait := neat.NewTrait()
	trait.Id = 1
	nodes := []*network.NNode{
		network.NewNNode(1, network.InputNeuron),
		network.NewNNode(4, network.HiddenNeuron),
	}

	gene, err := readPlainConnectionGene(strings.NewReader(geneStr), []*neat.Trait{trait}, nodes)
	require.NoError(t, err, "failed to read gene")

	assert.Equal(t, innovationNum, gene.InnovationNum)
	assert.Equal(t, mutNum, gene.MutationNum)
	assert.False(t, gene.IsEnabled)

	link := gene.Link
	require.NotNil(t, link)
	assert.Equal(t, traitId, link.Trait.Id)
	require.NotNil(t, link.InNode)
	assert.Equal(t, inNodeId, link.InNode.Id)
	require.NotNil(t, link.OutNode)
	assert.Equal(t, outNodeId, link.OutNode.Id)
	assert.Equal(t, weight, link.Weight)
	assert.False(t, link.IsRecurrent)
}

func TestPlainGenomeReader_ReadFile(t *testing.T) {
	genomePath := "../../data/xorstartgenes"
	genomeFile, err := os.Open(genomePath)
	require.NoError(t, err, "failed to read plain genome file")
	r, err := NewGenomeReader(genomeFile, PlainGenomeEncoding)
	require.NoError(t, err, "failed to create genome reader")
	genome, err := r.Read()
	require.NoError(t, err, "failed to read genome")
	require.NotNil(t, genome, "genome is empty")

	assert.Len(t, genome.Genes, 3, "wrong number of connection genes")
	assert.Len(t, genome.Nodes, 4, "wrong number of node genes")

	for i, n := range genome.Nodes {
		assert.Equal(t, i+1, n.Id, "Wrong NNode Id at: %d", i)
		if i == 0 {
			// BIAS
			assert.True(t, n.IsSensor(), "Wrong NNode type, SENSOR expected at: %d", i)
			assert.Equal(t, network.BiasNeuron, n.NeuronType, "Wrong NNode placement at: %d", i)
		}
		if i > 0 && i < 3 {
			// INPUT
			assert.True(t, n.IsSensor(), "Wrong NNode type, SENSOR expected at: %d", i)
			assert.Equal(t, network.InputNeuron, n.NeuronType, "Wrong NNode placement at: %d", i)
		}
		if i == 3 {
			// OUTPUT
			assert.True(t, n.IsNeuron(), "Wrong NNode type, NEURON expected at: %d", i)
			assert.Equal(t, network.OutputNeuron, n.NeuronType, "Wrong NNode placement at: %d", i)
		}
	}

	// check traits
	//
	require.Len(t, genome.Traits, 3, "wrong number of traits")
	for i, tr := range genome.Traits {
		assert.Equal(t, i+1, tr.Id, "Wrong Trait ID at: %d", i)
		require.Len(t, tr.Params, 8, "Wrong Trait's parameters length at: %d", i)

		expectedParams := []float64{float64(i+1) / 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
		assert.Equal(t, expectedParams, tr.Params, "Wrong Trait params read at: %d", i)
	}
}

func TestYAMLGenomeReader_Read(t *testing.T) {
	genomePath := "../../data/test_seed_genome.yml"
	genomeFile, err := os.Open(genomePath)
	require.NoError(t, err, "Failed to open genome file")
	r, err := NewGenomeReader(genomeFile, YAMLGenomeEncoding)
	require.NoError(t, err, "failed to create genome reader")
	genome, err := r.Read()
	require.NoError(t, err, "failed to read genome")
	require.NotNil(t, genome, "genome is empty")

	// Check nodes
	//
	require.Len(t, genome.Nodes, 14, "wrong number of nodes")
	nodes := []*network.NNode{
		{Id: 1, NeuronType: network.BiasNeuron, ActivationType: math.NullActivation},

		{Id: 2, NeuronType: network.InputNeuron, ActivationType: math.NullActivation},
		{Id: 3, NeuronType: network.InputNeuron, ActivationType: math.NullActivation},
		{Id: 4, NeuronType: network.InputNeuron, ActivationType: math.NullActivation},
		{Id: 5, NeuronType: network.InputNeuron, ActivationType: math.NullActivation},

		{Id: 6, NeuronType: network.OutputNeuron, ActivationType: math.SigmoidBipolarActivation},
		{Id: 7, NeuronType: network.OutputNeuron, ActivationType: math.SigmoidBipolarActivation},

		{Id: 8, NeuronType: network.HiddenNeuron, ActivationType: math.LinearActivation},
		{Id: 9, NeuronType: network.HiddenNeuron, ActivationType: math.LinearActivation},
		{Id: 10, NeuronType: network.HiddenNeuron, ActivationType: math.NullActivation},

		{Id: 11, NeuronType: network.HiddenNeuron, ActivationType: math.LinearActivation},
		{Id: 12, NeuronType: network.HiddenNeuron, ActivationType: math.LinearActivation},
		{Id: 13, NeuronType: network.HiddenNeuron, ActivationType: math.NullActivation},

		{Id: 14, NeuronType: network.HiddenNeuron, ActivationType: math.SignActivation},
	}
	for i, n := range nodes {
		assert.Equal(t, n.Id, genome.Nodes[i].Id, "at: %d", i)
		assert.Equal(t, n.ActivationType, genome.Nodes[i].ActivationType, "at: %d", i)
		assert.Equal(t, n.NeuronType, genome.Nodes[i].NeuronType, "at: %d", i)
	}

	// Check traits
	//
	require.Len(t, genome.Traits, 15, "wrong traits number")
	for i, tr := range genome.Traits {
		assert.Equal(t, i+1, tr.Id, "wrong trait ID at: %d", i)
		require.Len(t, tr.Params, 8, "Wrong Trait's parameters length at: %d", i)
		expectedParams := []float64{float64(i+1) / 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
		assert.Equal(t, expectedParams, tr.Params, "Wrong Trait params read at: %d", i)
	}

	// Check Genes
	//
	require.Len(t, genome.Genes, 15, "wrong number of connection genes")
	genes := []*Gene{
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[1], 0.0, nodes[0], nodes[5], false), 1, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[2], 0.0, nodes[1], nodes[5], false), 2, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[4], nodes[5], false), 3, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[4], 1.0, nodes[5], nodes[13], false), 4, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[4], 1.0, nodes[13], nodes[7], false), 5, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[1], nodes[8], false), 6, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[9], nodes[5], false), 7, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[3], nodes[11], false), 8, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 1.0, nodes[13], nodes[10], false), 9, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[12], nodes[5], false), 10, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[0], nodes[6], false), 11, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[2], nodes[6], false), 12, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[4], nodes[6], false), 13, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[9], nodes[6], false), 14, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(genome.Traits[3], 0.0, nodes[12], nodes[6], false), 15, 0, true),
	}
	for i, g := range genes {
		assert.Equal(t, g.InnovationNum, genome.Genes[i].InnovationNum, "at: %d", i)
		assert.Equal(t, g.IsEnabled, genome.Genes[i].IsEnabled, "at: %d", i)
		assert.Equal(t, g.MutationNum, genome.Genes[i].MutationNum, "at: %d", i)
		require.NotNil(t, genome.Genes[i].Link, "at: %d", i)
		assert.Equal(t, g.Link.Weight, genome.Genes[i].Link.Weight, "at: %d", i)
		require.NotNil(t, genome.Genes[i].Link.InNode, "at: %d", i)
		assert.Equal(t, g.Link.InNode.Id, genome.Genes[i].Link.InNode.Id, "at: %d", i)
		require.NotNil(t, genome.Genes[i].Link.OutNode, "at: %d", i)
		assert.Equal(t, g.Link.OutNode.Id, genome.Genes[i].Link.OutNode.Id, "at: %d", i)
	}

	// Check control genes
	//
	require.Len(t, genome.ControlGenes, 2, "wrong number of control genes")
	idCount := 8
	for i, g := range genome.ControlGenes {
		assert.EqualValues(t, i+16, g.InnovationNum, "Wrong innovation number at: %d", i)
		assert.Equal(t, 0.5+float64(i), g.MutationNum, "Wrong mutation number at: %d", i)
		assert.True(t, g.IsEnabled, "Wrong enabled value for module at: %d", i)

		require.NotNil(t, g.ControlNode, "at: %d", i)

		assert.Equal(t, i+15, g.ControlNode.Id, "Wrong control node ID at: %d", i)
		assert.Equal(t, math.MultiplyModuleActivation, g.ControlNode.ActivationType, "at: %d", i)
		assert.Equal(t, network.HiddenNeuron, g.ControlNode.NeuronType, "at: %d", i)

		require.Len(t, g.ControlNode.Incoming, 2, "at: %d", i)
		for j, l := range g.ControlNode.Incoming {
			assert.Equal(t, idCount, l.InNode.Id, "at: %d, %d", i, j)
			require.NotNil(t, l.OutNode, "at: %d, %d", i, j)
			assert.Equal(t, l.OutNode.Id, g.ControlNode.Id, "at: %d, %d", i, j)
			assert.Equal(t, 1.0, l.Weight, "at: %d, %d", i, j)
			idCount++
		}
		// check Outgoing InNode
		require.Len(t, g.ControlNode.Outgoing, 1, "at: %d", i)
		require.NotNil(t, g.ControlNode.Outgoing[0].InNode, "at: %d", i)
		assert.Equal(t, g.ControlNode.Id, g.ControlNode.Outgoing[0].InNode.Id, "at: %d", i)

		// check Outgoing OutNode
		require.NotNil(t, g.ControlNode.Outgoing[0].OutNode, "at: %d", i)
		assert.Equal(t, idCount, g.ControlNode.Outgoing[0].OutNode.Id, "at: %d", i)
		assert.Equal(t, 1.0, g.ControlNode.Outgoing[0].Weight, "at: %d", i)
		idCount++
	}
}
