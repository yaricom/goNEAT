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

func TestGenome_mutateAddLink(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	// Configuration
	context := &neat.Options{
		RecurOnlyProb:   0.5,
		NewLinkTries:    10,
		CompatThreshold: 0.5,
		PopSize:         1,
	}
	// The population with one organism
	pop := newPopulation()
	err := pop.spawn(gnome1, context)
	require.NoError(t, err, "failed to spawn population")

	// Create gnome phenotype
	_, err = gnome1.Genesis(1)
	require.NoError(t, err, "genesis failed")

	res, err := gnome1.mutateAddLink(pop, 1, context)
	require.NoError(t, err, "failed to add link")
	require.True(t, res, "New link not added")

	// one gene was added innovNum = 3 + 1
	assert.EqualValues(t, 4, pop.nextInnovNum, "wrong next innovation number of the population")
	assert.Len(t, pop.Innovations(), 1, "wrong number of innovations in population")
	assert.Len(t, gnome1.Genes, 4, "No new gene was added")
	gene := gnome1.Genes[3]
	assert.EqualValues(t, 4, gene.InnovationNum, "wrong innovation number of added gene")
	require.NotNil(t, gene.Link, "gene has no link")
	assert.True(t, gene.Link.IsRecurrent, "New gene must be recurrent, because only one NEURON node exists")

	// add more NEURONs
	context.RecurOnlyProb = 0.0
	nodes := []*network.NNode{
		{Id: 5, NeuronType: network.HiddenNeuron, ActivationType: math.SigmoidSteepenedActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
		{Id: 6, NeuronType: network.InputNeuron, ActivationType: math.SigmoidSteepenedActivation, Incoming: make([]*network.Link, 0), Outgoing: make([]*network.Link, 0)},
	}
	gnome1.addNodes(nodes)
	_, err = gnome1.Genesis(1) // do network genesis with new nodes added
	require.NoError(t, err, "genesis failed")

	res, err = gnome1.mutateAddLink(pop, 1, context)
	require.NoError(t, err, "failed to add link")
	require.True(t, res, "New link not added")

	// one gene was added innovNum = 4 + 1
	assert.EqualValues(t, 5, pop.nextInnovNum, "wrong next innovation number of the population")
	assert.Len(t, pop.Innovations(), 2, "wrong number of innovations in population")
	assert.Len(t, gnome1.Genes, 5, "No new gene was added")

	gene = gnome1.Genes[4]
	assert.EqualValues(t, 5, gene.InnovationNum, "wrong innovation number of added gene")
	require.NotNil(t, gene.Link, "gene has no link")
	assert.False(t, gene.Link.IsRecurrent, "New gene must not be recurrent, because conf.RecurOnlyProb = 0.0")
}

func TestGenome_mutateConnectSensors(t *testing.T) {
	// Test mutation with all inputs connected
	//
	gnome1 := buildTestGenome(1)
	// Create gnome phenotype
	_, err := gnome1.Genesis(1)
	require.NoError(t, err, "genesis failed")
	context := &neat.Options{}
	context.PopSize = 1
	// The population with one organism
	pop := newPopulation()
	err = pop.spawn(gnome1, context)
	require.NoError(t, err, "failed to spawn population")

	res, err := gnome1.mutateConnectSensors(pop, context)
	require.NoError(t, err, "failed to mutate")
	assert.False(t, res, "All inputs already connected - no mutation expected")

	// test with disconnected input
	//
	node := &network.NNode{
		Id:             5,
		NeuronType:     network.InputNeuron,
		ActivationType: math.SigmoidSteepenedActivation,
		Incoming:       make([]*network.Link, 0),
		Outgoing:       make([]*network.Link, 0)}
	gnome1.addNode(node)
	// Create gnome phenotype
	_, err = gnome1.Genesis(1)
	require.NoError(t, err, "genesis failed")
	res, err = gnome1.mutateConnectSensors(pop, context)
	require.NoError(t, err, "failed to mutate")
	assert.True(t, res, "Its expected for disconnected sensor to be connected now")
	assert.Len(t, gnome1.Genes, 4, "wrong number of genome genes")
	assert.Len(t, pop.Innovations(), 1, "wrong number of innovations")
	// one gene was added, expecting innovation + 1 (3+1)
	assert.EqualValues(t, 4, pop.nextInnovNum, "wrong innovation in population")
}

func TestGenome_mutateAddNode(t *testing.T) {
	//rand.Seed(42)
	gnome1 := buildTestGenome(1)

	// Create gnome phenotype
	_, err := gnome1.Genesis(1)
	require.NoError(t, err, "genesis failed")
	context := &neat.Options{
		NodeActivators:     []math.NodeActivationType{math.SigmoidSteepenedActivation},
		NodeActivatorsProb: []float64{1.0},
	}
	context.PopSize = 1
	// The population with one organism
	pop := newPopulation()
	err = pop.spawn(gnome1, context)
	require.NoError(t, err, "failed to spawn population")

	res, err := gnome1.mutateAddNode(pop, pop, context)
	require.NoError(t, err, "failed to mutate")
	require.True(t, res, "mutation failed")

	// two genes was added, expecting innovation + 2 (3+2)
	assert.EqualValues(t, 5, pop.nextInnovNum, "wrong next innovation number set for population")
	assert.Len(t, pop.Innovations(), 1, "wrong number of innovations")
	assert.Len(t, gnome1.Genes, 5, "wrong number of genes")
	require.Len(t, gnome1.Nodes, 5, "wrong number of nodes")

	addedNode := gnome1.Nodes[4]
	assert.Equal(t, 6, addedNode.Id, "New node has wrong ID")
	assert.Equal(t, math.SigmoidSteepenedActivation, addedNode.ActivationType, "wrong activation type")
}

func TestGenome_mutateLinkWeights(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	res, err := gnome1.mutateLinkWeights(0.5, 1.0, gaussianMutator)
	require.NoError(t, err, "failed to mutate")
	require.True(t, res, "mutation failed")

	for i, gn := range gnome1.Genes {
		// check that link weights are different from original ones (1.5, 2.5, 3.5)
		assert.NotEqual(t, float64(i)+1.5, gn.Link.ConnectionWeight, "Found not mutated gene: %s", gn)
	}
}

func TestGenome_mutateRandomTrait(t *testing.T) {
	gnome1 := buildTestGenome(1)
	// Configuration
	context := neat.Options{
		TraitMutationPower: 0.3,
		TraitParamMutProb:  0.5,
	}
	res, err := gnome1.mutateRandomTrait(&context)
	require.NoError(t, err, "failed to mutate")
	require.True(t, res, "mutation failed")

	mutationFound := false
	for _, tr := range gnome1.Traits {
		for pi, p := range tr.Params {
			if pi == 0 && p != float64(tr.Id)/10.0 {
				mutationFound = true
				break
			} else if pi > 0 && p != 0 {
				mutationFound = true
				break
			}
		}
		if mutationFound {
			break
		}
	}
	assert.True(t, mutationFound, "No mutation found in genome traits")
}

func TestGenome_mutateLinkTrait(t *testing.T) {
	gnome1 := buildTestGenome(1)

	res, err := gnome1.mutateLinkTrait(10)
	require.NoError(t, err, "failed to mutate")
	require.True(t, res, "mutation failed")

	mutationFound := false
	for i, gn := range gnome1.Genes {
		if gn.Link.Trait.Id != i+1 {
			mutationFound = true
			break
		}
	}
	assert.True(t, mutationFound, "No mutation found in gene links traits")
}

func TestGenome_mutateNodeTrait(t *testing.T) {
	gnome1 := buildTestGenome(1)

	// Add traits to nodes
	for i, nd := range gnome1.Nodes {
		if i < 3 {
			nd.Trait = gnome1.Traits[i]
		}
	}
	gnome1.Nodes[3].Trait = &neat.Trait{Id: 4, Params: []float64{0.4, 0, 0, 0, 0, 0, 0, 0}}

	res, err := gnome1.mutateNodeTrait(2)
	require.NoError(t, err, "failed to mutate")
	require.True(t, res, "mutation failed")

	mutationFound := false
	for i, nd := range gnome1.Nodes {
		if nd.Trait.Id != i+1 {
			mutationFound = true
			break
		}
	}
	assert.True(t, mutationFound, "No mutation found in nodes traits")
}

func TestGenome_mutateToggleEnable(t *testing.T) {
	gnome1 := buildTestGenome(1)
	// add extra connection gene from BIAS to OUT
	gene := NewConnectionGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2], gnome1.Nodes[3], false), 4, 0, true)
	gnome1.Genes = append(gnome1.Genes, gene)

	res, err := gnome1.mutateToggleEnable(50)
	require.NoError(t, err, "failed to mutate")
	require.True(t, res, "mutation failed")

	mutCount := 0
	for _, gn := range gnome1.Genes {
		if !gn.IsEnabled {
			mutCount++
		}
	}

	// in our genome only one connection gene can be disabled to not break the network (BIAS -> OUT) because
	// we added extra connection gene to link BIAS and OUT
	assert.Equal(t, 1, mutCount, "Wrong number of mutations found")
}

func TestGenome_mutateGeneReEnable(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	// add disabled extra connection gene from BIAS to OUT
	gene := NewConnectionGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2], gnome1.Nodes[3], false), 4, 0, false)
	gnome1.Genes = append(gnome1.Genes, gene)
	// disable one more gene
	gnome1.Genes[1].IsEnabled = false

	res, err := gnome1.mutateGeneReEnable()
	require.NoError(t, err, "failed to mutate")
	require.True(t, res, "mutation failed")

	// check that first encountered disabled gene was enabled and second disabled gene remain unchanged
	assert.True(t, gnome1.Genes[1].IsEnabled, "The first encountered gene should be enabled")
	assert.False(t, gnome1.Genes[3].IsEnabled, "The second disabled gene should still be disabled")
}
