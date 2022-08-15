package genetics

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v3/neat"
	"github.com/yaricom/goNEAT/v3/neat/math"
	"github.com/yaricom/goNEAT/v3/neat/network"
	"math/rand"
	"testing"
)

const gnomeStr = "genomestart 1\n" +
	"trait 1 0.1 0 0 0 0 0 0 0\n" +
	"trait 3 0.3 0 0 0 0 0 0 0\n" +
	"trait 2 0.2 0 0 0 0 0 0 0\n" +
	"node 1 0 1 1 NullActivation\n" + // SENSOR
	"node 2 0 1 1 NullActivation\n" + // SENSOR
	"node 3 0 1 3 SigmoidSteepenedActivation\n" + // BIAS
	"node 4 0 0 2 SigmoidSteepenedActivation\n" + // OUTPUT
	"gene 1 1 4 1.5 false 1 0 true\n" +
	"gene 2 2 4 2.5 false 2 0 true\n" +
	"gene 3 3 4 3.5 false 3 0 true\n" +
	"genomeend 1"

func buildTestGenome(id int) *Genome {
	traits := []*neat.Trait{
		{Id: 1, Params: []float64{0.1, 0, 0, 0, 0, 0, 0, 0}},
		{Id: 3, Params: []float64{0.3, 0, 0, 0, 0, 0, 0, 0}},
		{Id: 2, Params: []float64{0.2, 0, 0, 0, 0, 0, 0, 0}},
	}

	nodes := []*network.NNode{
		{Id: 1, NeuronType: network.InputNeuron, ActivationType: math.NullActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 2, NeuronType: network.InputNeuron, ActivationType: math.NullActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 3, NeuronType: network.BiasNeuron, ActivationType: math.SigmoidSteepenedActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 4, NeuronType: network.OutputNeuron, ActivationType: math.SigmoidSteepenedActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
	}

	genes := []*Gene{
		NewConnectionGene(network.NewLinkWithTrait(traits[0], 1.5, nodes[0], nodes[3], false), 1, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(traits[2], 2.5, nodes[1], nodes[3], false), 2, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(traits[1], 3.5, nodes[2], nodes[3], false), 3, 0, true),
	}

	return NewGenome(id, traits, nodes, genes)
}

func buildTestModularGenome(id int) *Genome {
	gnome := buildTestGenome(id)

	// append module with it's IO nodes
	ioNodes := []*network.NNode{
		{Id: 5, NeuronType: network.HiddenNeuron, ActivationType: math.LinearActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 6, NeuronType: network.HiddenNeuron, ActivationType: math.LinearActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 7, NeuronType: network.HiddenNeuron, ActivationType: math.NullActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
	}
	gnome.addNodes(ioNodes)

	// connect added nodes
	ioConnGenes := []*Gene{
		NewConnectionGene(network.NewLinkWithTrait(gnome.Traits[0], 1.5, gnome.Nodes[0], gnome.Nodes[4], false), 4, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(gnome.Traits[2], 2.5, gnome.Nodes[1], gnome.Nodes[5], false), 5, 0, true),
		NewConnectionGene(network.NewLinkWithTrait(gnome.Traits[1], 3.5, gnome.Nodes[6], gnome.Nodes[3], false), 6, 0, true),
	}
	gnome.Genes = append(gnome.Genes, ioConnGenes...)

	// add control gene
	controlNode := &network.NNode{
		Id: 8, NeuronType: network.HiddenNeuron,
		ActivationType: math.MultiplyModuleActivation,
	}
	controlNode.Incoming = []*network.Link{
		{ConnectionWeight: 1.0, InNode: ioNodes[0], OutNode: controlNode},
		{ConnectionWeight: 1.0, InNode: ioNodes[1], OutNode: controlNode},
	}
	controlNode.Outgoing = []*network.Link{
		{ConnectionWeight: 1.0, InNode: controlNode, OutNode: ioNodes[2]},
	}
	gnome.ControlGenes = []*MIMOControlGene{NewMIMOGene(controlNode, int64(7), 5.5, true)}
	return gnome
}

// Test create random genome
func TestGenome_NewGenomeRand(t *testing.T) {
	rand.Seed(42)
	newId, in, out, n := 1, 3, 2, 2
	opts := &neat.Options{
		CompatThreshold:    0.5,
		PopSize:            10,
		NodeActivators:     []math.NodeActivationType{math.GaussianBipolarActivation},
		NodeActivatorsProb: []float64{1.0},
	}

	gnome, err := newGenomeRand(newId, in, out, n, 5, false, 0.5, opts)
	require.NoError(t, err, "failed to create random genome")
	require.NotNil(t, gnome, "random genome expected")
	assert.Len(t, gnome.Nodes, in+n+out, "failed to create nodes")
	assert.True(t, len(gnome.Genes) >= in+n+out, "Failed to create genes")
}

// Test genesis
func TestGenome_Genesis(t *testing.T) {
	gnome := buildTestGenome(1)
	netId := 10

	net, err := gnome.Genesis(netId)
	require.NoError(t, err, "genesis failed")
	require.NotNil(t, net, "network expected")
	assert.Equal(t, netId, net.Id, "wrong network ID")
	assert.Equal(t, len(gnome.Nodes), net.NodeCount(), "wrong nodes count")
	assert.Equal(t, len(gnome.Genes), net.LinkCount(), "wrong links count")
}

func TestGenome_GenesisModular(t *testing.T) {
	gnome := buildTestModularGenome(1)
	netId := 10

	net, err := gnome.Genesis(netId)
	require.NoError(t, err, "genesis failed")
	require.NotNil(t, net, "network expected")
	assert.Equal(t, netId, net.Id, "wrong network ID")

	// check plain neuron nodes
	neuronNodesCount := len(gnome.Nodes) + len(gnome.ControlGenes)
	assert.Equal(t, neuronNodesCount, net.NodeCount(), "wrong nodes count")

	// find extra nodes and links due to MIMO control genes
	incoming, outgoing := 0, 0
	for _, cg := range gnome.ControlGenes {
		incoming += len(cg.ControlNode.Incoming)
		outgoing += len(cg.ControlNode.Outgoing)
	}
	// check connection genes
	connGenesCount := len(gnome.Genes) + incoming + outgoing
	assert.Equal(t, connGenesCount, net.LinkCount(), "wrong links count")
}

// Test duplicate
func TestGenome_Duplicate(t *testing.T) {
	gnome := buildTestGenome(1)

	newGnome, err := gnome.duplicate(2)
	require.NoError(t, err, "failed to duplicate")
	assert.Equal(t, 2, newGnome.Id)
	assert.Equal(t, len(gnome.Traits), len(newGnome.Traits), "wrong traits number")
	assert.Equal(t, len(gnome.Nodes), len(newGnome.Nodes), "wrong nodes number")
	assert.Equal(t, len(gnome.Genes), len(newGnome.Genes), "wrong genes number")

	equal, err := gnome.IsEqual(newGnome)
	assert.NoError(t, err)
	assert.True(t, equal, "equal genomes expected")
}

func TestGenome_DuplicateModular(t *testing.T) {
	gnome := buildTestModularGenome(1)

	newGnome, err := gnome.duplicate(2)
	require.NoError(t, err, "failed to duplicate")
	assert.Equal(t, 2, newGnome.Id)
	assert.Equal(t, len(gnome.Traits), len(newGnome.Traits), "wrong traits number")
	assert.Equal(t, len(gnome.Nodes), len(newGnome.Nodes), "wrong nodes number")
	assert.Equal(t, len(gnome.Genes), len(newGnome.Genes), "wrong genes number")
	assert.Equal(t, len(gnome.ControlGenes), len(newGnome.ControlGenes), "wrong control genes number")

	for i, cg := range newGnome.ControlGenes {
		ocg := gnome.ControlGenes[i]
		// check incoming connection genes
		assert.Equal(t, len(ocg.ControlNode.Incoming), len(cg.ControlNode.Incoming))
		for j, l := range cg.ControlNode.Incoming {
			ol := ocg.ControlNode.Incoming[j]
			assert.True(t, l.IsEqualGenetically(ol), "incoming are not equal genetically at: %d, id: %d",
				j, cg.ControlNode.Id)
		}
		// check outgoing connection genes
		assert.Equal(t, len(ocg.ControlNode.Outgoing), len(cg.ControlNode.Outgoing))
		for j, l := range cg.ControlNode.Outgoing {
			ol := ocg.ControlNode.Outgoing[j]
			assert.True(t, l.IsEqualGenetically(ol), "outgoing are not equal genetically at: %d, id: %d",
				j, cg.ControlNode.Id)
		}
	}
	equal, err := gnome.IsEqual(newGnome)
	assert.NoError(t, err)
	assert.True(t, equal, "equal genomes expected")
}

func TestGene_Verify(t *testing.T) {
	gnome := buildTestGenome(1)

	res, err := gnome.verify()
	require.NoError(t, err, "failed to verify")
	assert.True(t, res, "Verification failed")

	// Check gene missing input node failure
	//
	gene := NewGene(1.0, network.NewNNode(100, network.InputNeuron),
		network.NewNNode(4, network.OutputNeuron), false, 1, 1.0)
	gnome.Genes = append(gnome.Genes, gene)
	res, err = gnome.verify()
	require.EqualError(t, err, "missing input node of gene in the genome nodes list")
	assert.False(t, res, "Validation should fail")

	// Check gene missing output node failure
	//
	gnome = buildTestGenome(1)
	gene = NewGene(1.0, network.NewNNode(4, network.InputNeuron),
		network.NewNNode(400, network.OutputNeuron), false, 1, 1.0)
	gnome.Genes = append(gnome.Genes, gene)
	res, err = gnome.verify()
	require.EqualError(t, err, "missing output node of gene in the genome nodes list")
	assert.False(t, res, "Validation should fail")

	// Test duplicate genes failure
	//
	gnome = buildTestGenome(1)
	gnome.Genes = append(gnome.Genes, NewGene(1.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 1, 1.0))
	gnome.Genes = append(gnome.Genes, NewGene(1.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 1, 1.0))
	res, err = gnome.verify()
	require.Error(t, err, "error expected")
	assert.Contains(t, err.Error(), "duplicate genes found", "wrong error returned")
	assert.False(t, res, "Validation should fail")
}

func TestGenome_geneInsert(t *testing.T) {
	gnome := buildTestGenome(1)
	gnome.Genes = append(gnome.Genes, NewConnectionGene(network.NewLinkWithTrait(gnome.Traits[2], 5.5, gnome.Nodes[2], gnome.Nodes[3], false), 5, 0, false))
	gnome.geneInsert(NewConnectionGene(network.NewLinkWithTrait(gnome.Traits[2], 5.5, gnome.Nodes[2], gnome.Nodes[3], false), 4, 0, false))
	require.Equal(t, 5, len(gnome.Genes), "wrong genes number")

	for i, g := range gnome.Genes {
		if g.InnovationNum != int64(i+1) {
			t.Error("(g.InnovationNum != i + 1)", g.InnovationNum, i+1)
		}
	}
}
