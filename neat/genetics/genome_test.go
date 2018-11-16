package genetics

import (
	"testing"
	"strings"
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat"
	"bytes"
	"bufio"
	"math/rand"
)

const gnome_str = "genomestart 1\n" +
	"trait 1 0.1 0 0 0 0 0 0 0\n" +
	"trait 2 0.2 0 0 0 0 0 0 0\n" +
	"trait 3 0.3 0 0 0 0 0 0 0\n" +
	"node 1 0 1 1\n" +
	"node 2 0 1 1\n" +
	"node 3 0 1 3\n" +
	"node 4 0 0 2\n" +
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
		{Id:1, NeuronType: network.InputNeuron},
		{Id:2, NeuronType: network.InputNeuron},
		{Id:3, NeuronType: network.BiasNeuron},
		{Id:4, NeuronType: network.OutputNeuron},
	}

	genes := []*Gene{
		readPlainConnectionGene(strings.NewReader("1 1 4 1.5 false 1 0 true"), traits, nodes),
		readPlainConnectionGene(strings.NewReader("2 2 4 2.5 false 2 0 true"), traits, nodes),
		readPlainConnectionGene(strings.NewReader("3 3 4 3.5 false 3 0 true"), traits, nodes),
	}

	return NewGenome(id, traits, nodes, genes)
}

// Test write Genome
func TestGenome_WriteGenome(t *testing.T) {

	gnome := buildTestGenome(1)
	out_buf := bytes.NewBufferString("")
	gnome.Write(out_buf)

	_, g_str_r, err_g := bufio.ScanLines([]byte(gnome_str), true)
	_, o_str_r, err_o := bufio.ScanLines(out_buf.Bytes(), true)
	if err_g != nil || err_o != nil {
		t.Error("Failed to parse strings", err_o, err_g)
	}
	for i, gsr := range g_str_r {
		if gsr != o_str_r[i] {
			t.Error("Lines mismatch", gsr, o_str_r[i])
		}
	}
}

// Test create random genome
func TestGenome_NewGenomeRand(t *testing.T) {
	rand.Seed(42)
	new_id, in, out, n, nmax := 1, 3, 2, 2, 5
	recurrent := false
	link_prob := 0.5

	gnome := NewGenomeRand(new_id, in, out, n, nmax, recurrent, link_prob)

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

	net := gnome.genesis(net_id)
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

// Test duplicate
func TestGenome_Duplicate(t *testing.T) {
	gnome := buildTestGenome(1)

	new_gnome := gnome.duplicate(2)
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
	gnome2 := gnome1.duplicate(2)

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
	gnome1.genesis(1)

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
		readPlainNetworkNode(strings.NewReader("5 0 0 0"), gnome1.Traits),
		readPlainNetworkNode(strings.NewReader("6 0 0 1"), gnome1.Traits),
	}
	gnome1.Nodes = append(gnome1.Nodes, nodes...)
	gnome1.genesis(1) // do network genesis with new nodes added

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
	gnome1.genesis(1)

	context := neat.NeatContext{}

	res, err := gnome1.mutateConnectSensors(pop, &context)
	if err != nil {
		t.Error("err != nil", err)
		return
	}
	if res {
		t.Error("All inputs already connected - wrong return")
	}

	// adding disconnected input
	gnome1.Nodes = append(gnome1.Nodes,
		readPlainNetworkNode(strings.NewReader("5 0 1 1"), gnome1.Traits))
	// Create gnome phenotype
	gnome1.genesis(1)

	res, err = gnome1.mutateConnectSensors(pop, &context)
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
	gnome1.genesis(1)

	context := neat.NeatContext{}

	res, err := gnome1.mutateAddNode(pop, &context)
	if !res || err != nil {
		t.Error("Failed to add new node")
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
	gnome1.Genes = append(gnome1.Genes, readPlainConnectionGene(strings.NewReader("3 3 4 5.5 false 4 0 true"),
		gnome1.Traits, gnome1.Nodes))

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
	gnome1.Genes = append(gnome1.Genes, readPlainConnectionGene(strings.NewReader("3 3 4 5.5 false 4 0 false"),
		gnome1.Traits, gnome1.Nodes))

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
	gnome1.Genes = append(gnome1.Genes, readPlainConnectionGene(strings.NewReader("3 3 4 5.5 false 4 0 false"),
		gnome1.Traits, gnome1.Nodes))
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
		t.Error("len(gnome_child.Genes) != 3")
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4")
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3")
	}

	// check not equal gene pools
	gnome1.Genes = append(gnome1.Genes, readPlainConnectionGene(strings.NewReader("3 3 4 5.5 false 4 0 false"),
		gnome1.Traits, gnome1.Nodes))
	gnome2.Genes = append(gnome2.Genes, readPlainConnectionGene(strings.NewReader("3 2 4 5.5 true 4 0 false"),
		gnome1.Traits, gnome1.Nodes))
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
		t.Error("gnome_child.Nodes) != 4")
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3")
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
		t.Error("len(gnome_child.Genes) != 3")
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4")
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3")
	}

	// check not equal gene pools
	gnome1.Genes = append(gnome1.Genes, readPlainConnectionGene(strings.NewReader("3 3 4 5.5 false 4 0 false"),
		gnome1.Traits, gnome1.Nodes))
	gnome_child, err = gnome1.mateSinglepoint(gnome2, genomeid)
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

	gnome2.Genes = append(gnome1.Genes, readPlainConnectionGene(strings.NewReader("3 3 4 5.5 false 4 0 false"),
		gnome1.Traits, gnome1.Nodes))
	gnome2.Genes = append(gnome2.Genes, readPlainConnectionGene(strings.NewReader("3 2 4 5.5 true 4 0 false"),
		gnome1.Traits, gnome1.Nodes))
	gnome_child, err = gnome1.mateSinglepoint(gnome2, genomeid)
	if err != nil {
		t.Error(err)
	}
	if gnome_child == nil {
		t.Error("Failed to create child genome")
	}
	if len(gnome_child.Genes) != 4 {
		t.Error("len(gnome_child.Genes) != 4")
	}
	if len(gnome_child.Nodes) != 4 {
		t.Error("gnome_child.Nodes) != 4")
	}
	if len(gnome_child.Traits) != 3 {
		t.Error("len(gnome_child.Traits) != 3")
	}
}

func TestGenome_geneInsert(t *testing.T) {
	gnome := buildTestGenome(1)
	gnome.Genes = append(gnome.Genes, readPlainConnectionGene(strings.NewReader("3 3 4 5.5 false 5 0 false"),
		gnome.Traits, gnome.Nodes))

	gene := readPlainConnectionGene(strings.NewReader("3 3 4 5.5 false 4 0 false"), gnome.Traits, gnome.Nodes)
	genes := geneInsert(gnome.Genes, gene)
	if len(genes) != 5 {
		t.Error("len(genes) != 5", len(genes))
		return
	}

	for i, g := range genes {
		if g.InnovationNum != int64(i + 1) {
			t.Error("g.InnovationNum != i + 1", g.InnovationNum, i + 1)
			return
		}
	}
}