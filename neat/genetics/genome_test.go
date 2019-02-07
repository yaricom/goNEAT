package genetics

import (
	"testing"
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat"
	"math/rand"
	"github.com/yaricom/goNEAT/neat/utils"
)

const gnome_str = "genomestart 1\n" +
	"trait 1 0.1 0 0 0 0 0 0 0\n" +
	"trait 3 0.3 0 0 0 0 0 0 0\n" +
	"trait 2 0.2 0 0 0 0 0 0 0\n" +
	"node 1 0 1 1 NullActivation\n" +
	"node 2 0 1 1 NullActivation\n" +
	"node 3 0 1 3 SigmoidSteepenedActivation\n" +
	"node 4 0 0 2 SigmoidSteepenedActivation\n" +
	"gene 1 1 4 1.5 false 1 0 true\n" +
	"gene 2 2 4 2.5 false 2 0 true\n" +
	"gene 3 3 4 3.5 false 3 0 true\n" +
	"genomeend 1"

func buildTestGenome(id int) *Genome {
	traits := []*neat.Trait{
		{Id:1, Params:[]float64{0.1, 0, 0, 0, 0, 0, 0, 0}},
		{Id:3, Params:[]float64{0.3, 0, 0, 0, 0, 0, 0, 0}},
		{Id:2, Params:[]float64{0.2, 0, 0, 0, 0, 0, 0, 0}},
	}

	nodes := []*network.NNode{
		{Id:1, NeuronType: network.InputNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:2, NeuronType: network.InputNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:3, NeuronType: network.BiasNeuron, ActivationType: utils.SigmoidSteepenedActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:4, NeuronType: network.OutputNeuron, ActivationType: utils.SigmoidSteepenedActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
	}

	genes := []*Gene{
		newGene(network.NewLinkWithTrait(traits[0], 1.5, nodes[0], nodes[3], false), 1, 0, true),
		newGene(network.NewLinkWithTrait(traits[2], 2.5, nodes[1], nodes[3], false), 2, 0, true),
		newGene(network.NewLinkWithTrait(traits[1], 3.5, nodes[2], nodes[3], false), 3, 0, true),
	}

	return NewGenome(id, traits, nodes, genes)
}

func buildTestModularGenome(id int) *Genome {
	gnome := buildTestGenome(id)

	// append module with it's IO nodes
	io_nodes := []*network.NNode{
		{Id:5, NeuronType: network.HiddenNeuron, ActivationType: utils.LinearActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:6, NeuronType: network.HiddenNeuron, ActivationType: utils.LinearActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:7, NeuronType: network.HiddenNeuron, ActivationType: utils.NullActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
	}
	gnome.Nodes = append(gnome.Nodes, io_nodes ...)

	// connect added nodes
	io_conn_genes := []*Gene{
		newGene(network.NewLinkWithTrait(gnome.Traits[0], 1.5, gnome.Nodes[0], gnome.Nodes[4], false), 4, 0, true),
		newGene(network.NewLinkWithTrait(gnome.Traits[2], 2.5, gnome.Nodes[1], gnome.Nodes[5], false), 5, 0, true),
		newGene(network.NewLinkWithTrait(gnome.Traits[1], 3.5, gnome.Nodes[6], gnome.Nodes[3], false), 6, 0, true),
	}
	gnome.Genes = append(gnome.Genes, io_conn_genes ...)

	// add control gene
	c_node := &network.NNode{
		Id:8, NeuronType: network.HiddenNeuron,
		ActivationType: utils.MultiplyModuleActivation,
	}
	c_node.Incoming = []*network.Link{
		{Weight:1.0, InNode:io_nodes[0], OutNode:c_node},
		{Weight:1.0, InNode:io_nodes[1], OutNode:c_node},
	}
	c_node.Outgoing = []*network.Link{
		{Weight:1.0, InNode:c_node, OutNode:io_nodes[2]},
	}
	gnome.ControlGenes = []*MIMOControlGene{NewMIMOGene(c_node, int64(7), 5.5, true)}
	return gnome
}

// Test create random genome
func TestGenome_NewGenomeRand(t *testing.T) {
	rand.Seed(42)
	new_id, in, out, n, nmax := 1, 3, 2, 2, 5
	recurrent := false
	link_prob := 0.5

	gnome := newGenomeRand(new_id, in, out, n, nmax, recurrent, link_prob)

	if gnome == nil {
		t.Error("Failed to create random genome")
	}
	if len(gnome.Nodes) != in + n + out {
		t.Error("len(gnome.Nodes) != in + nmax + out", len(gnome.Nodes), in + n + out)
	}
	if len(gnome.Genes) < in + n + out {
		t.Error("Failed to create genes", len(gnome.Genes))
	}

	//for _, g := range gnome.Genes {
	//	t.Log(g)
	//}
}

// Test genesis
func TestGenome_Genesis(t *testing.T) {
	gnome := buildTestGenome(1)

	net_id := 10

	net, err := gnome.Genesis(net_id)
	if err != nil {
		t.Error(err)
		return
	}
	if net == nil {
		t.Error("Failed to do network genesis")
	}
	if net.Id != net_id {
		t.Error("net.Id != net_id", net.Id)
	}
	if net.NodeCount() != len(gnome.Nodes) {
		t.Error("net.NodeCount() != len(nodes)", net.NodeCount(), len(gnome.Nodes))
	}
	if net.LinkCount() != len(gnome.Genes) {
		t.Error("net.LinkCount() != len(genes)", net.LinkCount(), len(gnome.Genes))
	}
}

func TestGenome_GenesisModular(t *testing.T) {
	gnome := buildTestModularGenome(1)

	net_id := 10

	net, err := gnome.Genesis(net_id)
	if err != nil {
		t.Error(err)
		return
	}
	if net == nil {
		t.Error("Failed to do modular network genesis")
	}
	if net.Id != net_id {
		t.Error("net.Id != net_id", net.Id)
	}

	// check plain neuron nodes
	neuron_nodes_count := len(gnome.Nodes) + len(gnome.ControlGenes)
	if net.NodeCount() != neuron_nodes_count {
		t.Error("net.NodeCount() != neuron_nodes_count", net.NodeCount(), neuron_nodes_count)
	}

	// find extra nodes and links due to MIMO control genes
	incoming, outgoing := 0, 0
	for _, cg := range gnome.ControlGenes {
		incoming += len(cg.ControlNode.Incoming)
		outgoing += len(cg.ControlNode.Outgoing)
	}
	// check connection genes
	conn_genes_count := len(gnome.Genes) + incoming + outgoing
	if net.LinkCount() != conn_genes_count {
		t.Error("net.LinkCount() != conn_genes_count", net.LinkCount(), conn_genes_count)
	}
}

// Test duplicate
func TestGenome_Duplicate(t *testing.T) {
	gnome := buildTestGenome(1)

	new_gnome, err := gnome.duplicate(2)
	if err != nil {
		t.Error(err)
		return
	}
	if new_gnome.Id != 2 {
		t.Error("new_gnome.Id != 2", new_gnome.Id)
	}

	if len(new_gnome.Traits) != len(gnome.Traits) {
		t.Error("len(new_gnome.Traits) != len(gnome.Traits)", len(new_gnome.Traits), len(gnome.Traits))
	}
	if len(new_gnome.Nodes) != len(gnome.Nodes) {
		t.Error("len(new_gnome.Nodes) != len(gnome.Nodes)", len(new_gnome.Nodes), len(gnome.Nodes))
	}
	if len(new_gnome.Genes) != len(gnome.Genes) {
		t.Error("len(new_gnome.Genes) != len(gnome.Genes)", len(new_gnome.Genes), len(gnome.Genes))
	}

	equal, err := gnome.IsEqual(new_gnome)
	if !equal {
		t.Error(err)
	}
}

func TestGenome_DuplicateModular(t *testing.T) {
	gnome := buildTestModularGenome(1)

	new_gnome, err := gnome.duplicate(2)
	if err != nil {
		t.Error(err)
		return
	}
	if new_gnome.Id != 2 {
		t.Error("new_gnome.Id != 2", new_gnome.Id)
	}

	if len(new_gnome.Traits) != len(gnome.Traits) {
		t.Error("len(new_gnome.Traits) != len(gnome.Traits)", len(new_gnome.Traits), len(gnome.Traits))
	}
	if len(new_gnome.Nodes) != len(gnome.Nodes) {
		t.Error("len(new_gnome.Nodes) != len(gnome.Nodes)", len(new_gnome.Nodes), len(gnome.Nodes))
	}
	if len(new_gnome.Genes) != len(gnome.Genes) {
		t.Error("len(new_gnome.Genes) != len(gnome.Genes)", len(new_gnome.Genes), len(gnome.Genes))
	}
	if len(new_gnome.ControlGenes) != len(gnome.ControlGenes) {
		t.Error("len(new_gnome.ControlGenes) != len(gnome.ControlGenes", len(new_gnome.ControlGenes), len(gnome.ControlGenes))
	}
	for i, cg := range new_gnome.ControlGenes {
		ocg := gnome.ControlGenes[i]
		if len(cg.ControlNode.Incoming) != len(ocg.ControlNode.Incoming) {
			t.Error("len(cg.ControlNode.Incoming) != len(ocg.ControlNode.Incoming) at:", cg.String())
		}
		for i, l := range cg.ControlNode.Incoming {
			ol := ocg.ControlNode.Incoming[i]
			if !l.IsEqualGenetically(ol) {
				t.Error("!l.IsEqualGenetically(ol) incoming at: ", cg.ControlNode.Id)
			}
		}
		if len(cg.ControlNode.Outgoing) != len(ocg.ControlNode.Outgoing) {
			t.Error("len(cg.ControlNode.Outgoing) != len(ocg.ControlNode.Outgoing) at:", cg.String())
		}
		for i, l := range cg.ControlNode.Outgoing {
			ol := ocg.ControlNode.Outgoing[i]
			if !l.IsEqualGenetically(ol) {
				t.Error("!l.IsEqualGenetically(ol) outgoing at: ", cg.ControlNode.Id)
			}
		}
	}
	equal, err := gnome.IsEqual(new_gnome)
	if !equal {
		t.Error(err)
	}
}

func TestGene_Verify(t *testing.T) {
	gnome := buildTestGenome(1)

	res, err := gnome.verify()

	if !res {
		t.Error("Verification failed", err)
	}

	if err != nil {
		t.Error("err != nil", err)
	}

	// Test gene error
	new_gene := NewGene(1.0, network.NewNNode(100, network.InputNeuron),
		network.NewNNode(101, network.OutputNeuron), false, 1, 1.0)
	gnome.Genes = append(gnome.Genes, new_gene)
	res, err = gnome.verify()
	if res {
		t.Error("Validation should fail")
	}
	if err == nil {
		t.Error("Error is missing")
	}

	// Test duplicate genes
	gnome = buildTestGenome(1)
	gnome.Genes = append(gnome.Genes, NewGene(1.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 1, 1.0))
	gnome.Genes = append(gnome.Genes, NewGene(1.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 1, 1.0))
	res, err = gnome.verify()
	if res {
		t.Error("Validation should fail")
	}
	if err == nil {
		t.Error("Error is missing")
	}

}

func TestGenome_Compatibility_Linear(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)

	// Configuration
	conf := neat.NeatContext{
		DisjointCoeff:0.5,
		ExcessCoeff:0.5,
		MutdiffCoeff:0.5,
		GenCompatMethod:0,
	}

	// Test fully compatible
	comp := gnome1.compatibility(gnome2, &conf)
	if comp != 0 {
		t.Error("comp != 0 ", comp)
	}

	// Test incompatible
	gnome2.Genes = append(gnome2.Genes, NewGene(1.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 10, 1.0))
	comp = gnome1.compatibility(gnome2, &conf)
	if comp != 0.5 {
		t.Error("comp != 0.5", comp)
	}

	gnome2.Genes = append(gnome2.Genes, NewGene(2.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 5, 1.0))
	comp = gnome1.compatibility(gnome2, &conf)
	if comp != 1 {
		t.Error("comp != 1", comp)
	}

	gnome2.Genes[1].MutationNum = 6.0
	comp = gnome1.compatibility(gnome2, &conf)
	if comp != 2 {
		t.Error("comp != 2", comp)
	}
}

func TestGenome_Compatibility_Fast(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)

	// Configuration
	conf := neat.NeatContext{
		DisjointCoeff:0.5,
		ExcessCoeff:0.5,
		MutdiffCoeff:0.5,
		GenCompatMethod:1,
	}

	// Test fully compatible
	comp := gnome1.compatibility(gnome2, &conf)
	if comp != 0 {
		t.Error("comp != 0 ", comp)
	}

	// Test incompatible
	gnome2.Genes = append(gnome2.Genes, NewGene(1.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 10, 1.0))
	comp = gnome1.compatibility(gnome2, &conf)
	if comp != 0.5 {
		t.Error("comp != 0.5", comp)
	}

	gnome2.Genes = append(gnome2.Genes, NewGene(2.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(1, network.OutputNeuron), false, 5, 1.0))
	comp = gnome1.compatibility(gnome2, &conf)
	if comp != 1 {
		t.Error("comp != 1", comp)
	}

	gnome2.Genes[1].MutationNum = 6.0
	comp = gnome1.compatibility(gnome2, &conf)
	if comp != 2 {
		t.Error("comp != 2", comp)
	}
}

func TestGenome_Compatibility_Duplicate(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2, err := gnome1.duplicate(2)
	if err != nil {
		t.Error(err)
		return
	}

	// Configuration
	conf := neat.NeatContext{
		DisjointCoeff:0.5,
		ExcessCoeff:0.5,
		MutdiffCoeff:0.5,
	}

	// Test fully compatible
	comp := gnome1.compatibility(gnome2, &conf)
	if comp != 0 {
		t.Error("comp != 0 ", comp)
	}
}

func TestGenome_mutateAddLink(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	// Configuration
	conf := neat.NeatContext{
		RecurOnlyProb:0.5,
		NewLinkTries:10,
		CompatThreshold:0.5,
	}
	// The population (DUMMY)
	pop := newPopulation()
	pop.nextInnovNum = int64(4)
	// Create gnome phenotype
	gnome1.Genesis(1)

	res, err := gnome1.mutateAddLink(pop, &conf)
	if !res {
		t.Error("New link not added:", err)
		return
	}
	if err != nil {
		t.Error("err != nil", err)
	}
	if pop.nextInnovNum != 5 {
		t.Error("pop.currInnovNum != 5", pop.nextInnovNum)
	}
	if len(pop.Innovations) != 1 {
		t.Error("len(pop.Innovations) != 1", len(pop.Innovations))
	}
	if len(gnome1.Genes) != 4 {
		t.Error("No new gene was added")
	}
	if gnome1.Genes[3].InnovationNum != 5 {
		t.Error("gnome1.Genes[3].InnovationNum != 5", gnome1.Genes[3].InnovationNum)
	}
	if !gnome1.Genes[3].Link.IsRecurrent {
		t.Error("New gene must be recurrent, because only one NEURON node exists")
	}

	// add more NEURONs
	conf.RecurOnlyProb = 0.0
	nodes := []*network.NNode{
		{Id:5, NeuronType: network.HiddenNeuron, ActivationType: utils.SigmoidSteepenedActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
		{Id:6, NeuronType: network.InputNeuron, ActivationType: utils.SigmoidSteepenedActivation, Incoming:make([]*network.Link, 0), Outgoing:make([]*network.Link, 0)},
	}
	gnome1.Nodes = append(gnome1.Nodes, nodes...)
	gnome1.Genesis(1) // do network genesis with new nodes added

	res, err = gnome1.mutateAddLink(pop, &conf)
	if !res {
		t.Error("New link not added:", err)
	}
	if err != nil {
		t.Error("err != nil", err)
	}
	if pop.nextInnovNum != 6 {
		t.Error("pop.currInnovNum != 6", pop.nextInnovNum)
	}
	if len(pop.Innovations) != 2 {
		t.Error("len(pop.Innovations) != 2", len(pop.Innovations))
	}
	if len(gnome1.Genes) != 5 {
		t.Error("No new gene was added")
	}
	if gnome1.Genes[4].Link.IsRecurrent {
		t.Error("New gene must not be recurrent, because conf.RecurOnlyProb = 0.0")
	}
	//t.Log(gnome1.Genes[4])
}

func TestGenome_mutateConnectSensors(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	// The population (DUMMY)
	pop := newPopulation()
	// Create gnome phenotype
	gnome1.Genesis(1)

	context := neat.NewNeatContext()

	res, err := gnome1.mutateConnectSensors(pop, context)
	if err != nil {
		t.Error("err != nil", err)
		return
	}
	if res {
		t.Error("All inputs already connected - wrong return")
	}

	// adding disconnected input
	node := &network.NNode{
		Id:5,
		NeuronType: network.InputNeuron,
		ActivationType: utils.SigmoidSteepenedActivation,
		Incoming:make([]*network.Link, 0),
		Outgoing:make([]*network.Link, 0)}
	gnome1.Nodes = append(gnome1.Nodes, node)
	// Create gnome phenotype
	gnome1.Genesis(1)

	res, err = gnome1.mutateConnectSensors(pop, context)
	if err != nil {
		t.Error("err != nil", err)
		return
	}
	if !res {
		t.Error("Its expected for disconnected sensor to be connected now")
	}
	if len(gnome1.Genes) != 4 {
		t.Error("len(gnome1.Genes) != 4", len(gnome1.Genes))
	}
	if len(pop.Innovations) != 1 {
		t.Error("len(pop.Innovations) != 1", len(pop.Innovations))
	}
}

func TestGenome_mutateAddNode(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)

	// The population (DUMMY)
	pop := newPopulation()
	// Create gnome phenotype
	gnome1.Genesis(1)

	context := neat.NewNeatContext()

	res, err := gnome1.mutateAddNode(pop, context)
	if !res || err != nil {
		t.Error("Failed to add new node:", err)
		return
	}
	if pop.nextInnovNum != 2 {
		t.Error("pop.currInnovNum != 2", pop.nextInnovNum)
	}
	if len(pop.Innovations) != 1 {
		t.Error("len(pop.Innovations) != 1", len(pop.Innovations))
	}
	if len(gnome1.Genes) != 5 {
		t.Error("No new gene was added")
	}
	if len(gnome1.Nodes) != 5 {
		t.Error("No new node was added to genome")
	}
	if gnome1.Nodes[0].Id != 1 {
		t.Error("New node has wrong ID:", gnome1.Nodes[0].Id)
	}
}

func TestGenome_mutateLinkWeights(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	// Configuration
	conf := neat.NeatContext{
		WeightMutPower:0.5,
	}

	res, err := gnome1.mutateLinkWeights(conf.WeightMutPower, 1.0, gaussianMutator)
	if !res || err != nil {
		t.Error("Failed to mutate link weights")
	}
	for i, gn := range gnome1.Genes {
		if gn.Link.Weight == float64(i) + 1.5 {
			t.Error("Found not mutrated gene:", gn)
		}
	}
}

func TestGenome_mutateRandomTrait(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	// Configuration
	conf := neat.NeatContext{
		TraitMutationPower:0.3,
		TraitParamMutProb:0.5,
	}
	res, err := gnome1.mutateRandomTrait(&conf)
	if !res || err != nil {
		t.Error("Failed to mutate random trait")
	}
	mutation_found := false
	for _, tr := range gnome1.Traits {
		for pi, p := range tr.Params {
			if pi == 0 && p != float64(tr.Id) / 10.0 {
				mutation_found = true
				break
			} else if pi > 0 && p != 0 {
				mutation_found = true
				break
			}
		}
		if mutation_found {
			break
		}
	}
	if !mutation_found {
		t.Error("No mutation found in genome traits")
	}
}

func TestGenome_mutateLinkTrait(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)

	res, err := gnome1.mutateLinkTrait(10)
	if !res || err != nil {
		t.Error("Failed to mutate link trait")
	}
	mutation_found := false
	for i, gn := range gnome1.Genes {
		if gn.Link.Trait.Id != i + 1 {
			mutation_found = true
			break
		}
	}
	if !mutation_found {
		t.Error("No mutation found in gene links traits")
	}
}

func TestGenome_mutateNodeTrait(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)

	// Add traits to nodes
	for i, nd := range gnome1.Nodes {
		if i < 3 {
			nd.Trait = gnome1.Traits[i]
		}
	}
	gnome1.Nodes[3].Trait = &neat.Trait{Id:4, Params: []float64{0.4, 0, 0, 0, 0, 0, 0, 0}}

	res, err := gnome1.mutateNodeTrait(2)
	if !res || err != nil {
		t.Error("Failed to mutate node trait")
	}
	mutation_found := false
	for i, nd := range gnome1.Nodes {
		if nd.Trait.Id != i + 1 {
			mutation_found = true
			break
		}
	}
	if !mutation_found {
		t.Error("No mutation found in nodes traits")
	}
}

func TestGenome_mutateToggleEnable(t *testing.T) {
	rand.Seed(41)
	gnome1 := buildTestGenome(1)
	gene := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2], gnome1.Nodes[3], false), 4, 0, true)
	gnome1.Genes = append(gnome1.Genes, gene)

	res, err := gnome1.mutateToggleEnable(5)
	if !res || err != nil {
		t.Error("Failed to mutate toggle genes")
	}
	mut_count := 0
	for _, gn := range gnome1.Genes {
		if !gn.IsEnabled {
			mut_count++
		}
	}
	if mut_count != 1 {
		t.Errorf("Wrong number of mutations found: %d", mut_count)
	}
}

func TestGenome_mutateGeneReenable(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gene := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2], gnome1.Nodes[3], false), 4, 0, false)
	gnome1.Genes = append(gnome1.Genes, gene)

	gnome1.Genes[1].IsEnabled = false
	res, err := gnome1.mutateGeneReenable()
	if !res || err != nil {
		t.Error("Failed to mutate toggle genes")
	}
	if !gnome1.Genes[1].IsEnabled {
		t.Error("The first gene should be enabled")
	}
	if gnome1.Genes[3].IsEnabled {
		t.Error("The second gen should be still disabled")
	}
}

func TestGenome_mateMultipoint(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)

	genomeid := 3
	fitness1, fitness2 := 1.0, 2.3

	gnome_child, err := gnome1.mateMultipoint(gnome2, genomeid, fitness1, fitness2)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}

	if len(gnome_child.Genes) != 3 {
		t.Error("len(gnome_child.Genes) != 3")
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4")
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3")
	}

	// check not equal gene pools
	gene := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2], gnome1.Nodes[3], false), 4, 0, true)
	gnome1.Genes = append(gnome1.Genes, gene)
	fitness1, fitness2 = 15.0, 2.3
	gnome_child, err = gnome1.mateMultipoint(gnome2, genomeid, fitness1, fitness2)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}
	if len(gnome_child.Genes) != 3 {
		t.Error("len(gnome_child.Genes) != 3")
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4")
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3")
	}
}

func TestGenome_mateMultipointModular(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestModularGenome(2)

	genomeid := 3
	fitness1, fitness2 := 1.0, 2.3

	gnome_child, err := gnome1.mateMultipoint(gnome2, genomeid, fitness1, fitness2)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}

	if len(gnome_child.Genes) != 6 {
		t.Error("len(gnome_child.Genes) != 6", len(gnome_child.Genes))
	}
	if len(gnome_child.Nodes) != 7 {
		t.Error("len(gnome_child.Nodes) != 7", len(gnome_child.Nodes))
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3", len(gnome_child.Traits))
	}
	if len(gnome_child.ControlGenes) != 1 {
		t.Error("len(gnome_child.ControlGenes) != 1", len(gnome_child.ControlGenes))
	}
}

func TestGenome_mateMultipointAvg(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)

	genomeid := 3
	fitness1, fitness2 := 1.0, 2.3
	gnome_child, err := gnome1.mateMultipointAvg(gnome2, genomeid, fitness1, fitness2)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}
	if len(gnome_child.Genes) != 3 {
		t.Error("len(gnome_child.Genes) != 3", len(gnome_child.Genes))
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4", len(gnome_child.Nodes))
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3", len(gnome_child.Traits))
	}

	// check not size equal gene pools
	gene := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2], gnome1.Nodes[3], false), 4, 0, false)
	gnome1.Genes = append(gnome1.Genes, gene)
	gene2 := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[1], gnome1.Nodes[3], true), 4, 0, false)
	gnome2.Genes = append(gnome2.Genes, gene2)

	fitness1, fitness2 = 15.0, 2.3
	gnome_child, err = gnome1.mateMultipointAvg(gnome2, genomeid, fitness1, fitness2)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}
	if len(gnome_child.Genes) != 4 {
		t.Error("len(gnome_child.Genes) != 4", len(gnome_child.Genes))
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4", len(gnome_child.Nodes))
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3", len(gnome_child.Traits))
	}
}

func TestGenome_mateMultipointAvgModular(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestModularGenome(2)

	genomeid := 3
	fitness1, fitness2 := 1.0, 2.3

	gnome_child, err := gnome1.mateMultipointAvg(gnome2, genomeid, fitness1, fitness2)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}

	if len(gnome_child.Genes) != 6 {
		t.Error("len(gnome_child.Genes) != 6", len(gnome_child.Genes))
	}
	if len(gnome_child.Nodes) != 7 {
		t.Error("len(gnome_child.Nodes) != 7", len(gnome_child.Nodes))
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3", len(gnome_child.Traits))
	}
	if len(gnome_child.ControlGenes) != 1 {
		t.Error("len(gnome_child.ControlGenes) != 1", len(gnome_child.ControlGenes))
	}
}

func TestGenome_mateSinglepoint(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestGenome(2)

	genomeid := 3
	gnome_child, err := gnome1.mateSinglepoint(gnome2, genomeid)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}
	if len(gnome_child.Genes) != 3 {
		t.Error("len(gnome_child.Genes) != 3", len(gnome_child.Genes))
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4", len(gnome_child.Nodes))
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3", len(gnome_child.Traits))
	}

	// check not size equal gene pools
	gene := newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2], gnome1.Nodes[3], false), 4, 0, false)
	gnome1.Genes = append(gnome1.Genes, gene)
	gnome_child, err = gnome1.mateSinglepoint(gnome2, genomeid)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}
	if len(gnome_child.Genes) != 3 {
		t.Error("len(gnome_child.Genes) != 3", len(gnome_child.Genes))
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4", len(gnome_child.Nodes))
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3", len(gnome_child.Traits))
	}

	// set second Genome genes to first + one more
	gnome2.Genes = append(gnome1.Genes, newGene(network.NewLinkWithTrait(gnome1.Traits[2], 5.5, gnome1.Nodes[2], gnome1.Nodes[3], false), 4, 0, false))
	// append additional gene
	gnome2.Genes = append(gnome2.Genes, newGene(network.NewLinkWithTrait(gnome2.Traits[2], 5.5, gnome2.Nodes[1], gnome2.Nodes[3], true), 4, 0, false))

	gnome_child, err = gnome1.mateSinglepoint(gnome2, genomeid)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}
	if len(gnome_child.Genes) != 4 {
		t.Error("len(gnome_child.Genes) != 4", len(gnome_child.Genes))
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4", len(gnome_child.Nodes))
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3", len(gnome_child.Traits))
	}
}

func TestGenome_mateSinglepointModular(t *testing.T) {
	rand.Seed(42)
	gnome1 := buildTestGenome(1)
	gnome2 := buildTestModularGenome(2)

	genomeid := 3

	gnome_child, err := gnome1.mateSinglepoint(gnome2, genomeid)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}

	if len(gnome_child.Genes) != 6 {
		t.Error("len(gnome_child.Genes) != 6", len(gnome_child.Genes))
	}
	if len(gnome_child.Nodes) != 7 {
		t.Error("len(gnome_child.Nodes) != 7", len(gnome_child.Nodes))
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3", len(gnome_child.Traits))
	}
	if len(gnome_child.ControlGenes) != 1 {
		t.Error("len(gnome_child.ControlGenes) != 1", len(gnome_child.ControlGenes))
	}
}

func TestGenome_geneInsert(t *testing.T) {
	gnome := buildTestGenome(1)
	gnome.Genes = append(gnome.Genes, newGene(network.NewLinkWithTrait(gnome.Traits[2], 5.5, gnome.Nodes[2], gnome.Nodes[3], false), 5, 0, false))
	genes := geneInsert(gnome.Genes, newGene(network.NewLinkWithTrait(gnome.Traits[2], 5.5, gnome.Nodes[2], gnome.Nodes[3], false), 4, 0, false))
	if len(genes) != 5 {
		t.Error("len(genes) != 5", len(genes))
		return
	}

	for i, g := range genes {
		if g.InnovationNum != int64(i + 1) {
			t.Error("(g.InnovationNum != i + 1)", g.InnovationNum, i + 1)
		}
	}
}