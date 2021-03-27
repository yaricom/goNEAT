package genetics

import (
	"errors"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"github.com/yaricom/goNEAT/v2/neat/utils"
	"io"
	"math"
	"math/rand"
	"reflect"
)

// A Genome is the primary source of genotype information used to create  a phenotype.
// It contains 3 major constituents:
// 	1) A Vector of Traits
// 	2) A List of NNodes pointing to a Trait from (1)
// 	3) A List of Genes with Links that point to Traits from (1)
// 	4) A List of MIMO Control Genes with Links to different genome modules
//
// (1) Reserved parameter space for future use.
// (2) NNode specifications.
// (3) Is the primary source of innovation in the evolutionary Genome.
// (4) Control genes allows to receive inputs from multiple independent genome modules and output processed signal to the
//     multitude of output locations
//
// Each Gene in (3) has a marker telling when it arose historically. Thus, these Genes can be used to speciate the
// population, and the list of Genes provide an evolutionary history of innovation and link-building.
type Genome struct {
	// The genome ID
	Id int
	// The parameters conglomerations
	Traits []*neat.Trait
	// List of NNodes for the Network
	Nodes []*network.NNode
	// List of innovation-tracking genes
	Genes []*Gene
	// List of MIMO control genes
	ControlGenes []*MIMOControlGene

	// Allows Genome to be matched with its Network
	Phenotype *network.Network
}

// Constructor which takes full genome specs and puts them into the new one
func NewGenome(id int, t []*neat.Trait, n []*network.NNode, g []*Gene) *Genome {
	return &Genome{
		Id:     id,
		Traits: t,
		Nodes:  n,
		Genes:  g,
	}
}

// Constructs new modular genome
func NewModularGenome(id int, t []*neat.Trait, n []*network.NNode, g []*Gene, mimoG []*MIMOControlGene) *Genome {
	return &Genome{
		Id:           id,
		Traits:       t,
		Nodes:        n,
		Genes:        g,
		ControlGenes: mimoG,
	}
}

// This special constructor creates a Genome with in inputs, out outputs, n out of maxHidden hidden units, and random
// connectivity.  If rec is true then recurrent connections will be included. The last input is a bias
// link_prob is the probability of a link. The created genome is not modular.
func newGenomeRand(newId, in, out, n, maxHidden int, recurrent bool, linkProb float64) *Genome {
	totalNodes := in + out + maxHidden
	matrixDim := totalNodes * totalNodes
	// The connection matrix which will be randomized
	cm := make([]bool, matrixDim) //Dimension the connection matrix

	// No nodes above this number for this genome
	maxNode := in + n
	firstOutput := totalNodes - out + 1

	// Create a dummy trait (this is for future expansion of the system)
	newTrait := neat.NewTrait()
	newTrait.Id = 1
	newTrait.Params = make([]float64, neat.NumTraitParams)

	// Create empty genome
	gnome := Genome{
		Id:     newId,
		Traits: []*neat.Trait{newTrait},
		Nodes:  make([]*network.NNode, 0),
		Genes:  make([]*Gene, 0),
	}

	// Step through the connection matrix, randomly assigning bits
	for count := 0; count < matrixDim; count++ {
		cm[count] = rand.Float64() < linkProb
	}

	// Build the input nodes
	for i := 1; i <= in; i++ {
		var newNode *network.NNode
		if i < in {
			newNode = network.NewNNode(i, network.InputNeuron)
		} else {
			newNode = network.NewNNode(i, network.BiasNeuron)
		}
		newNode.Trait = newTrait
		gnome.Nodes = append(gnome.Nodes, newNode)
	}

	// Build the hidden nodes
	for i := in + 1; i <= in+n; i++ {
		newNode := network.NewNNode(i, network.HiddenNeuron)
		newNode.Trait = newTrait
		gnome.Nodes = append(gnome.Nodes, newNode)
	}

	// Build the output nodes
	for i := firstOutput; i <= totalNodes; i++ {
		newNode := network.NewNNode(i, network.OutputNeuron)
		newNode.Trait = newTrait
		gnome.Nodes = append(gnome.Nodes, newNode)
	}

	//
	//    i i i n n n n n n n n n n n n n n n n . . . . . . . . o o o o
	//    |                                   |                 ^     |
	//    |<----------- max_node ------------>|                 |     |
	//    |                                                     |     |
	//    |<-----------------------total_nodes -----------------|---->|
	//                                                          |
	//                                                          |
	//     first_output ----------------------------------------+
	//
	//

	var inNode, outNode *network.NNode
	// Step through the connection matrix, creating connection genes
	count := 0
	var flagRecurrent bool
	for col := 1; col <= totalNodes; col++ {
		for row := 1; row <= totalNodes; row++ {
			// Only try to create a link if it is in the matrix and not leading into a sensor
			if cm[count] && col > in &&
				(col <= maxNode || col >= firstOutput) &&
				(row <= maxNode || row >= firstOutput) {

				// If it's recurrent, create the connection (gene) no matter what
				createGene := true
				if col > row {
					flagRecurrent = false
				} else {
					flagRecurrent = true
					if !recurrent {
						// skip recurrent connections
						createGene = false
					}
				}

				// Introduce new connection (gene) into genome
				if createGene {
					// Retrieve in_node and out_node
					for i := 0; i < len(gnome.Nodes) && (inNode == nil || outNode == nil); i++ {
						nodeId := gnome.Nodes[i].Id
						if nodeId == row {
							inNode = gnome.Nodes[i]
						}
						if nodeId == col {
							outNode = gnome.Nodes[i]
						}
					}

					// Create the gene
					weight := float64(utils.RandSign()) * rand.Float64()
					gene := NewGeneWithTrait(newTrait, weight, inNode, outNode, flagRecurrent, int64(count), weight)

					//Add the gene to the genome
					gnome.Genes = append(gnome.Genes, gene)
				}

			}

			count++ //increment counter
			// reset nodes
			inNode, outNode = nil, nil
		}
	}
	return &gnome
}

// Reads Genome from reader
func ReadGenome(ir io.Reader, id int) (*Genome, error) {
	// stub for backward compatibility
	// the new implementations should use GenomeReader to decode genome data in variety of formats
	r, err := NewGenomeReader(ir, PlainGenomeEncoding)
	if err != nil {
		return nil, err
	}
	gnome, err := r.Read()
	return gnome, err
}

// Writes this genome into provided writer
func (g *Genome) Write(w io.Writer) error {
	// stub for backward compatibility
	// the new implementations should use GenomeWriter to decode genome data in variety of formats
	wr, err := NewGenomeWriter(w, PlainGenomeEncoding)
	if err == nil {
		err = wr.WriteGenome(g)
	}

	return err
}

// Stringer
func (g *Genome) String() string {
	str := "GENOME START\nNodes:\n"
	for _, n := range g.Nodes {
		nodeType := ""
		switch n.NeuronType {
		case network.InputNeuron:
			nodeType = "I"
		case network.OutputNeuron:
			nodeType = "O"
		case network.BiasNeuron:
			nodeType = "B"
		case network.HiddenNeuron:
			nodeType = "H"
		}
		str += fmt.Sprintf("\t%s%s \n", nodeType, n)
	}
	str += "Genes:\n"
	for _, gn := range g.Genes {
		str += fmt.Sprintf("\t%s\n", gn)
	}
	str += "Traits:\n"
	for _, t := range g.Traits {
		str += fmt.Sprintf("\t%s\n", t)
	}
	str += "GENOME END"
	return str
}

// Return # of non-disabled genes
func (g *Genome) Extrons() int {
	total := 0
	for _, gene := range g.Genes {
		if gene.IsEnabled {
			total++
		}
	}
	return total
}

// Tests if given genome is equal to this one genetically and phenotypically. This method will check that both genomes has the same traits, nodes and genes.
// If mismatch detected the error will be returned with mismatch details.
func (g *Genome) IsEqual(og *Genome) (bool, error) {
	if len(g.Traits) != len(og.Traits) {
		return false, errors.New(fmt.Sprintf("traits count mismatch: %d != %d",
			len(g.Traits), len(og.Traits)))
	}
	for i, tr := range og.Traits {
		if !reflect.DeepEqual(tr, g.Traits[i]) {
			return false, errors.New(
				fmt.Sprintf("traits mismatch, expected: %s, but found: %s", tr, g.Traits[i]))
		}
	}

	if len(g.Nodes) != len(og.Nodes) {
		return false, errors.New(fmt.Sprintf("nodes count mismatch: %d != %d",
			len(g.Nodes), len(og.Nodes)))
	}
	for i, nd := range og.Nodes {
		if !reflect.DeepEqual(nd, g.Nodes[i]) {
			return false, errors.New(
				fmt.Sprintf("node mismatch, expected: %s\nfound: %s", nd, g.Nodes[i]))
		}
	}

	if len(g.Genes) != len(og.Genes) {
		return false, errors.New(fmt.Sprintf("genes count mismatch: %d != %d",
			len(g.Genes), len(og.Genes)))
	}
	for i, gen := range og.Genes {
		if !reflect.DeepEqual(gen, g.Genes[i]) {
			return false, errors.New(
				fmt.Sprintf("gene mismatch, expected: %s\nfound: %s", gen, g.Genes[i]))
		}
	}

	if len(g.ControlGenes) != len(og.ControlGenes) {
		return false, errors.New(fmt.Sprintf("control genes count mismatch: %d != %d",
			len(g.ControlGenes), len(og.ControlGenes)))
	}
	for i, cg := range og.ControlGenes {
		if !reflect.DeepEqual(cg, g.ControlGenes[i]) {
			return false, errors.New(
				fmt.Sprintf("control gene mismatch, expected: %s\nfound: %s", cg, g.ControlGenes[i]))
		}
	}

	return true, nil
}

// Return id of final NNode in Genome
func (g *Genome) getLastNodeId() (int, error) {
	if len(g.Nodes) == 0 {
		return -1, errors.New("genome has no nodes")
	}
	id := g.Nodes[len(g.Nodes)-1].Id
	// check control genes
	for _, cg := range g.ControlGenes {
		if cg.ControlNode.Id > id {
			id = cg.ControlNode.Id
		}
	}
	return id, nil
}

// Return innovation number of last gene in Genome + 1
func (g *Genome) getNextGeneInnovNum() (int64, error) {
	innNum := int64(0)
	// check connection genes
	if len(g.Genes) > 0 {
		innNum = g.Genes[len(g.Genes)-1].InnovationNum
	} else {
		return -1, errors.New("genome has no Genes")
	}
	// check control genes if any
	if len(g.ControlGenes) > 0 {
		cInnNum := g.ControlGenes[len(g.ControlGenes)-1].InnovationNum
		if cInnNum > innNum {
			innNum = cInnNum
		}
	}
	return innNum + int64(1), nil
}

// Returns true if this Genome already includes provided node
func (g *Genome) hasNode(node *network.NNode) bool {
	if node == nil {
		return false
	}
	if id, _ := g.getLastNodeId(); node.Id > id {
		return false // not found
	}
	for _, n := range g.Nodes {
		if n.Id == node.Id {
			return true
		}
	}
	return false
}

// Returns true if this Genome already includes provided gene
func (g *Genome) hasGene(gene *Gene) bool {
	if inn, _ := g.getNextGeneInnovNum(); gene.InnovationNum >= inn {
		return false
	}

	// Find genetically equal link in this genome to the provided gene
	for _, g := range g.Genes {
		if g.Link.IsEqualGenetically(gene.Link) {
			return true
		}
	}
	return false
}

// Generate a Network phenotype from this Genome with specified id
func (g *Genome) Genesis(netId int) (*network.Network, error) {
	// Inputs and outputs will be collected here for the network.
	// All nodes are collected in an all_list -
	// this is useful for network traversing routines
	inList := make([]*network.NNode, 0)
	outList := make([]*network.NNode, 0)
	allList := make([]*network.NNode, 0)

	var newNode *network.NNode
	// Create the network nodes
	for _, n := range g.Nodes {
		newNode = network.NewNNodeCopy(n, n.Trait)

		// Check for input or output designation of node
		if n.NeuronType == network.InputNeuron || n.NeuronType == network.BiasNeuron {
			inList = append(inList, newNode)
		} else if n.NeuronType == network.OutputNeuron {
			outList = append(outList, newNode)
		}

		// Keep track of all nodes in one place for convenience
		allList = append(allList, newNode)

		// Have the node specifier point to the node it generated
		n.PhenotypeAnalogue = newNode
	}

	if len(g.Genes) == 0 {
		return nil, errors.New("network built without GENES; the result can be unpredictable")
	}

	if len(outList) == 0 {
		return nil, fmt.Errorf("network without OUTPUTS; the result can be unpredictable. Genome: %s", g)
	}

	var inNode, outNode *network.NNode
	var curLink, newLink *network.Link
	// Create the links by iterating through the genes
	for _, gn := range g.Genes {
		// Only create the link if the gene is enabled
		if gn.IsEnabled {
			curLink = gn.Link
			inNode = curLink.InNode.PhenotypeAnalogue
			outNode = curLink.OutNode.PhenotypeAnalogue

			// NOTE: This line could be run through a recurrence check if desired
			// (no need to in the current implementation of NEAT)
			newLink = network.NewLinkWithTrait(curLink.Trait, curLink.Weight, inNode, outNode, curLink.IsRecurrent)

			// Add link to the connected nodes
			outNode.Incoming = append(outNode.Incoming, newLink)
			inNode.Outgoing = append(inNode.Outgoing, newLink)
		}
	}

	var newNet *network.Network
	if len(g.ControlGenes) == 0 {
		// Create the new network
		newNet = network.NewNetwork(inList, outList, allList, netId)
	} else {
		// Create MIMO control genes
		cNodes := make([]*network.NNode, 0)
		for _, cg := range g.ControlGenes {
			// Only process enabled genes
			if cg.IsEnabled {
				newCopyNode := network.NewNNodeCopy(cg.ControlNode, cg.ControlNode.Trait)

				// connect inputs
				for _, l := range cg.ControlNode.Incoming {
					inNode = l.InNode.PhenotypeAnalogue
					outNode = newCopyNode
					newLink = network.NewLink(l.Weight, inNode, outNode, false)
					// only incoming to control node
					outNode.Incoming = append(outNode.Incoming, newLink)
				}

				// connect outputs
				for _, l := range cg.ControlNode.Outgoing {
					inNode = newCopyNode
					outNode = l.OutNode.PhenotypeAnalogue
					newLink = network.NewLink(l.Weight, inNode, outNode, false)
					// only outgoing from control node
					inNode.Outgoing = append(inNode.Outgoing, newLink)
				}

				// store control node
				cNodes = append(cNodes, newCopyNode)
			}
		}
		newNet = network.NewModularNetwork(inList, outList, allList, cNodes, netId)
	}

	// Attach genotype and phenotype together:
	// genotype points to owner phenotype (new_net)
	g.Phenotype = newNet

	return newNet, nil
}

// Duplicate this Genome to create a new one with the specified id
func (g *Genome) duplicate(newId int) (*Genome, error) {

	// Duplicate the traits
	traitsDup := make([]*neat.Trait, 0)
	for _, tr := range g.Traits {
		newTrait := neat.NewTraitCopy(tr)
		traitsDup = append(traitsDup, newTrait)
	}

	// Duplicate NNodes
	nodesDup := make([]*network.NNode, 0)
	for _, nd := range g.Nodes {
		// First, find the duplicate of the trait that this node points to
		assocTrait := nd.Trait
		if assocTrait != nil {
			assocTrait = traitWithId(assocTrait.Id, traitsDup)
		}
		newNode := network.NewNNodeCopy(nd, assocTrait)

		nodesDup = append(nodesDup, newNode)
	}

	// Duplicate Genes
	genesDup := make([]*Gene, 0)
	for _, gn := range g.Genes {
		// First find the nodes connected by the gene's link
		inNode := nodeWithId(gn.Link.InNode.Id, nodesDup)
		if inNode == nil {
			return nil, errors.New(
				fmt.Sprintf("incoming node: %d not found for gene %s",
					gn.Link.InNode.Id, gn.String()))
		}
		outNode := nodeWithId(gn.Link.OutNode.Id, nodesDup)
		if outNode == nil {
			return nil, errors.New(
				fmt.Sprintf("outgoing node: %d not found for gene %s",
					gn.Link.OutNode.Id, gn.String()))
		}

		// Find the duplicate of trait associated with this gene
		assocTrait := gn.Link.Trait
		if assocTrait != nil {
			assocTrait = traitWithId(assocTrait.Id, traitsDup)
		}

		gene := NewGeneCopy(gn, assocTrait, inNode, outNode)
		genesDup = append(genesDup, gene)
	}

	if len(g.ControlGenes) == 0 {
		// If no MIMO control genes return plain genome
		return NewGenome(newId, traitsDup, nodesDup, genesDup), nil
	} else {
		// Duplicate MIMO Control Genes and build modular genome
		controlGenesDup := make([]*MIMOControlGene, 0)
		for _, cg := range g.ControlGenes {
			// duplicate control node
			controlNode := cg.ControlNode
			// find duplicate of trait associated with control node
			assocTrait := controlNode.Trait
			if assocTrait != nil {
				assocTrait = traitWithId(assocTrait.Id, traitsDup)
			}
			nodeCopy := network.NewNNodeCopy(controlNode, assocTrait)
			// add incoming links
			for _, l := range controlNode.Incoming {
				inNode := nodeWithId(l.InNode.Id, nodesDup)
				if inNode == nil {
					return nil, fmt.Errorf("incoming node: %d not found for control node: %d",
						l.InNode.Id, controlNode.Id)
				}
				newInLink := network.NewLinkCopy(l, inNode, nodeCopy)
				nodeCopy.Incoming = append(nodeCopy.Incoming, newInLink)
			}

			// add outgoing links
			for _, l := range controlNode.Outgoing {
				outNode := nodeWithId(l.OutNode.Id, nodesDup)
				if outNode == nil {
					return nil, fmt.Errorf("outgoing node: %d not found for control node: %d",
						l.InNode.Id, controlNode.Id)
				}
				newOutLink := network.NewLinkCopy(l, nodeCopy, outNode)
				nodeCopy.Outgoing = append(nodeCopy.Outgoing, newOutLink)
			}

			// add MIMO control gene
			geneCopy := NewMIMOGeneCopy(cg, nodeCopy)
			controlGenesDup = append(controlGenesDup, geneCopy)
		}

		return NewModularGenome(newId, traitsDup, nodesDup, genesDup, controlGenesDup), nil
	}
}

// For debugging: A number of tests can be run on a genome to check its integrity.
// Note: Some of these tests do not indicate a bug, but rather are meant to be used to detect specific system states.
func (g *Genome) verify() (bool, error) {
	if len(g.Genes) == 0 {
		return false, errors.New("genome has no Genes")
	}
	if len(g.Nodes) == 0 {
		return false, errors.New("genome has no Nodes")
	}
	if len(g.Traits) == 0 {
		return false, errors.New("genome has no Traits")
	}

	// Check each gene's nodes
	for _, gn := range g.Genes {
		inNode := gn.Link.InNode
		outNode := gn.Link.OutNode
		inputFound, outFound := false, false
		for i := 0; i < len(g.Nodes) && (!inputFound || !outFound); i++ {
			if inNode.Id == g.Nodes[i].Id {
				inputFound = true
			}
			if outNode.Id == g.Nodes[i].Id {
				outFound = true
			}
		}

		// check results
		if !inputFound {
			return false, errors.New("missing input node of gene in the genome nodes")
		}
		if !outFound {
			return false, errors.New("missing output node of gene in the genome nodes")
		}
	}

	// Check for NNodes being out of order
	lastId := 0
	for _, n := range g.Nodes {
		if n.Id < lastId {
			return false, errors.New("nodes out of order in genome")
		}
		lastId = n.Id
	}

	// Make sure there are no duplicate genes
	for _, gn := range g.Genes {
		for _, gn2 := range g.Genes {
			if gn != gn2 && gn.Link.IsEqualGenetically(gn2.Link) {
				return false, fmt.Errorf("duplicate genes found. %s == %s", gn, gn2)
			}
		}
	}
	// Check for 2 disables in a row
	// Note: Again, this is not necessarily a bad sign
	if len(g.Nodes) > 500 {
		disabled := false
		for _, gn := range g.Genes {
			if gn.IsEnabled == false && disabled {
				return false, errors.New("two gene disables in a row")
			}
			disabled = !gn.IsEnabled
		}
	}
	return true, nil
}

// Inserts a NNode into a given ordered list of NNodes in ascending order by NNode ID
func nodeInsert(nodes []*network.NNode, n *network.NNode) []*network.NNode {
	if n == nil {
		neat.WarnLog("GENOME: attempting to insert NIL node into genome nodes, recovered")
		return nodes
	}
	index := len(nodes)
	// quick insert at the end or beginning (we assume that nodes is already ordered)
	if index == 0 || n.Id >= nodes[index-1].Id {
		// append last
		nodes = append(nodes, n)
		return nodes
	} else if n.Id <= nodes[0].Id {
		// insert first
		index = 0
	}
	// find split index
	for i := index - 1; i >= 0; i-- {
		if n.Id == nodes[i].Id {
			index = i
			break
		} else if n.Id > nodes[i].Id {
			index = i + 1 // previous
			break
		}
	}
	first := make([]*network.NNode, index+1)
	copy(first, nodes[0:index])
	first[index] = n
	second := nodes[index:]

	nodes = append(first, second...)
	return nodes
}

// Inserts a new gene that has been created through a mutation in the
// *correct order* into the list of genes in the genome, i.e. ordered by innovation number ascending
func geneInsert(genes []*Gene, g *Gene) []*Gene {
	if g == nil {
		neat.WarnLog("GENOME: attempting to insert NIL gere into genome genes, recovered")
		return genes
	}

	index := len(genes) // to make sure that greater IDs appended at the end
	// quick insert at the end or beginning (we assume that nodes is already ordered)
	if index == 0 || g.InnovationNum >= genes[index-1].InnovationNum {
		// append last
		genes = append(genes, g)
		return genes
	} else if g.InnovationNum <= genes[0].InnovationNum {
		// insert first
		index = 0
	}
	// find split index
	for i := index - 1; i >= 0; i-- {
		if g.InnovationNum == genes[i].InnovationNum {
			index = i
			break
		} else if g.InnovationNum > genes[i].InnovationNum {
			index = i + 1 // previous
			break
		}
	}

	first := make([]*Gene, index+1)
	copy(first, genes[0:index])
	first[index] = g
	second := genes[index:]
	genes = append(first, second...)
	return genes
}

/* ******* MUTATORS ******* */

// Mutate the genome by adding connections to disconnected sensors (input, bias type neurons).
// The reason this mutator is important is that if we can start NEAT with some inputs disconnected,
// then we can allow NEAT to decide which inputs are important.
// This process has two good effects:
// 	(1) You can start minimally even in problems with many inputs and
// 	(2) you don't need to know a priori what the important features of the domain are.
// If all sensors already connected than do nothing.
func (g *Genome) mutateConnectSensors(pop *Population, _ *neat.NeatContext) (bool, error) {

	if len(g.Genes) == 0 {
		return false, errors.New("genome has no genes")
	}

	// Find all the sensors and outputs
	sensors := make([]*network.NNode, 0)
	outputs := make([]*network.NNode, 0)
	for _, n := range g.Nodes {
		if n.IsSensor() {
			sensors = append(sensors, n)
		} else {
			outputs = append(outputs, n)
		}
	}

	// Find not connected sensors if any
	disconnectedSensors := make([]*network.NNode, 0)
	for _, sensor := range sensors {
		connected := false

		// iterate over all genes and count number of output connections from given sensor
		for _, gene := range g.Genes {
			if gene.Link.InNode.Id == sensor.Id {
				connected = true
				break
			}
		}

		if !connected {
			// store found disconnected sensor
			disconnectedSensors = append(disconnectedSensors, sensor)
		}
	}

	// if all sensors are connected - stop
	if len(disconnectedSensors) == 0 {
		return false, nil
	}

	// pick randomly from disconnected sensors
	sensor := disconnectedSensors[rand.Intn(len(disconnectedSensors))]
	// add new links to chosen sensor, avoiding redundancy
	linkAdded := false
	for _, output := range outputs {
		found := false
		for _, gene := range g.Genes {
			if gene.Link.InNode == sensor && gene.Link.OutNode == output {
				found = true
				break
			}
		}

		if !found {
			var gene *Gene
			// Check to see if this innovation already occurred in the population
			innovationFound := false
			for _, inn := range pop.Innovations {
				if inn.innovationType == newLinkInnType &&
					inn.InNodeId == sensor.Id &&
					inn.OutNodeId == output.Id &&
					inn.IsRecurrent == false {

					gene = NewGeneWithTrait(g.Traits[inn.NewTraitNum], inn.NewWeight,
						sensor, output, false, inn.InnovationNum, 0)

					innovationFound = true
					break
				}
			}

			// The innovation is totally novel
			if !innovationFound {
				// Choose a random trait
				traitNum := rand.Intn(len(g.Traits))
				// Choose the new weight
				newWeight := float64(utils.RandSign()) * rand.Float64() * 10.0
				// read next innovation id
				nextInnovId := pop.getNextInnovationNumberAndIncrement()

				// Create the new gene
				gene = NewGeneWithTrait(g.Traits[traitNum], newWeight, sensor, output,
					false, nextInnovId, newWeight)

				// Add the innovation for created link
				newInnovation := NewInnovationForLink(sensor.Id, output.Id, nextInnovId,
					newWeight, traitNum)
				pop.addInnovationSynced(newInnovation)
			} else if gene != nil && g.hasGene(gene) {
				// The gene for already occurred innovation already in this genome.
				// This may happen as result of parent genome mutation in current epoch which is
				// repeated in the child after parent's genome transferred to child during mating
				neat.InfoLog(
					fmt.Sprintf("GENOME: Connect sensors innovation found [%t] in the same genome [%d] for gene: %s\n%s",
						innovationFound, g.Id, gene, g))
				return false, nil
			}

			// Now add the new Gene to the Genome
			if gene != nil {
				g.Genes = geneInsert(g.Genes, gene)
				linkAdded = true
			}
		}
	}
	return linkAdded, nil
}

// Mutate the genome by adding a new link between two random NNodes,
// if NNodes are already connected, keep trying conf.NewLinkTries times
func (g *Genome) mutateAddLink(pop *Population, context *neat.NeatContext) (bool, error) {
	// If the phenotype does not exist, exit on false, print error
	// Note: This should never happen - if it does there is a bug
	if g.Phenotype == nil {
		return false, errors.New("attempt to add link to genome with no phenotype")
	} else if len(g.Nodes) == 0 {
		return false, errors.New("genome has no nodes to be connected by new link")
	}

	nodesLen := len(g.Nodes)

	// Decide whether to make link recurrent
	doRecur := false
	if rand.Float64() < context.RecurOnlyProb {
		doRecur = true
	}

	// Find the first non-sensor so that the to-node won't look at sensors as possible destinations
	firstNonSensor := 0
	for _, n := range g.Nodes {
		if n.IsSensor() {
			firstNonSensor++
		} else {
			break
		}
	}

	// Made attempts to find an unconnected pair
	tryCount := 0

	// Iterate over nodes and try to add new link
	var node1, node2 *network.NNode
	found := false
	for tryCount < context.NewLinkTries {
		nodeNum1, nodeNum2 := 0, 0
		if doRecur {
			// 50% of prob to decide create a recurrent link (node X to node X)
			// 50% of a normal link (node X to node Y)
			loopRecur := false
			if rand.Float64() > 0.5 {
				loopRecur = true
			}
			if loopRecur {
				nodeNum1 = firstNonSensor + rand.Intn(nodesLen-firstNonSensor) // only NON SENSOR
				nodeNum2 = nodeNum1
			} else {
				for nodeNum1 == nodeNum2 {
					nodeNum1 = rand.Intn(nodesLen)
					nodeNum2 = firstNonSensor + rand.Intn(nodesLen-firstNonSensor) // only NON SENSOR
				}
			}
		} else {
			for nodeNum1 == nodeNum2 {
				nodeNum1 = rand.Intn(nodesLen)
				nodeNum2 = firstNonSensor + rand.Intn(nodesLen-firstNonSensor) // only NON SENSOR
			}
		}

		// get corresponding nodes
		node1 = g.Nodes[nodeNum1]
		node2 = g.Nodes[nodeNum2]

		// See if a link already exists  ALSO STOP AT END OF GENES!!!!
		linkExists := false
		if node2.IsSensor() {
			// Don't allow SENSORS to get input
			linkExists = true
		} else {
			for _, gene := range g.Genes {
				if gene.Link.InNode.Id == node1.Id &&
					gene.Link.OutNode.Id == node2.Id &&
					gene.Link.IsRecurrent == doRecur {
					// link already exists
					linkExists = true
					break
				}

			}
		}

		if !linkExists {
			// These are used to avoid getting stuck in an infinite loop checking for recursion
			// Note that we check for recursion to control the frequency of adding recurrent links rather
			// than to prevent any particular kind of error
			thresh := nodesLen * nodesLen
			count := 0
			recurFlag := g.Phenotype.IsRecurrent(node1.PhenotypeAnalogue, node2.PhenotypeAnalogue, &count, thresh)

			// NOTE: A loop doesn't really matter - just debug output it
			if count > thresh {
				if neat.LogLevel == neat.LogLevelDebug {
					neat.DebugLog(
						fmt.Sprintf("GENOME: LOOP DETECTED DURING A RECURRENCY CHECK -> "+
							"node in: %s <-> node out: %s", node1.PhenotypeAnalogue, node2.PhenotypeAnalogue))
				}
			}

			// Make sure it finds the right kind of link (recurrent or not)
			if (!recurFlag && doRecur) || (recurFlag && !doRecur) {
				tryCount++
			} else {
				// The open link found
				tryCount = context.NewLinkTries
				found = true
			}
		} else {
			tryCount++
		}

	}
	// Continue only if an open link was found and corresponding nodes was set
	if node1 != nil && node2 != nil && found {
		var gene *Gene
		// Check to see if this innovation already occurred in the population
		innovationFound := false
		for _, inn := range pop.Innovations {
			// match the innovation in the innovations list
			if inn.innovationType == newLinkInnType &&
				inn.InNodeId == node1.Id &&
				inn.OutNodeId == node2.Id &&
				inn.IsRecurrent == doRecur {

				// Create new gene
				gene = NewGeneWithTrait(g.Traits[inn.NewTraitNum], inn.NewWeight, node1, node2, doRecur, inn.InnovationNum, 0)

				innovationFound = true
				break
			}
		}
		// The innovation is totally novel
		if !innovationFound {
			// Choose a random trait
			traitNum := rand.Intn(len(g.Traits))
			// Choose the new weight
			newWeight := float64(utils.RandSign()) * rand.Float64() * 10.0
			// read next innovation id
			nextInnovId := pop.getNextInnovationNumberAndIncrement()

			// Create the new gene
			gene = NewGeneWithTrait(g.Traits[traitNum], newWeight, node1, node2,
				doRecur, nextInnovId, newWeight)

			// Add the innovation
			innovation := NewInnovationForRecurrentLink(node1.Id, node2.Id, nextInnovId,
				newWeight, traitNum, doRecur)
			pop.addInnovationSynced(innovation)
		} else if gene != nil && g.hasGene(gene) {
			// The gene for already occurred innovation already in this genome.
			// This may happen as result of parent genome mutation in current epoch which is
			// repeated in the child after parent's genome transferred to child during mating
			neat.InfoLog(
				fmt.Sprintf("GENOME: Mutate add link innovation found [%t] in the same genome [%d] for gene: %s\n%s",
					innovationFound, g.Id, gene, g))
			return false, nil
		}

		// sanity check
		if gene != nil && gene.Link.InNode.Id == gene.Link.OutNode.Id && !doRecur {
			neat.WarnLog(fmt.Sprintf("Recurent link created when recurency is not enabled: %s", gene))
			return false, fmt.Errorf("wrong gene created\n%s", g)
		}

		// Now add the new Gene to the Genome
		if gene != nil {
			g.Genes = geneInsert(g.Genes, gene)
		}
	}

	return found, nil
}

// This mutator adds a node to a Genome by inserting it in the middle of an existing link between two nodes.
// This broken link will be disabled and now represented by two links with the new node between them.
// The innovations list from population is used to compare the innovation with other innovations in the list and see
// whether they match. If they do, the same innovation numbers will be assigned to the new genes. If a disabled link
// is chosen, then the method just exits with false.
func (g *Genome) mutateAddNode(pop *Population, context *neat.NeatContext) (bool, error) {
	if len(g.Genes) == 0 {
		return false, nil // it's possible to have such a network without any link
	}

	// First, find a random gene already in the genome
	found := false
	var gene *Gene

	// For a very small genome, we need to bias splitting towards older links to avoid a "chaining" effect which is likely
	// to occur when we keep splitting between the same two nodes over and over again (on newer and newer connections)
	if len(g.Genes) < 15 {
		for _, gn := range g.Genes {
			// Now randomize which gene is chosen.
			if gn.IsEnabled && gn.Link.InNode.NeuronType != network.BiasNeuron && rand.Float32() >= 0.3 {
				gene = gn
				found = true
				break
			}
		}
	} else {
		tryCount := 0
		// Alternative uniform random choice of genes. When the genome is not tiny, it is safe to choose randomly.
		for tryCount < 20 && !found {
			geneNum := rand.Intn(len(g.Genes))
			gene = g.Genes[geneNum]
			if gene.IsEnabled && gene.Link.InNode.NeuronType != network.BiasNeuron {
				found = true
			}
			tryCount++
		}
	}
	if !found || gene == nil {
		// Failed to find appropriate gene
		return false, nil
	}

	gene.IsEnabled = false

	// Extract the link
	link := gene.Link
	// Extract the weight
	oldWeight := link.Weight
	// Get the old link's trait
	trait := link.Trait

	// Extract the nodes
	inNode, outNode := link.InNode, link.OutNode
	if inNode == nil || outNode == nil {
		return false, fmt.Errorf("Genome:mutateAddNode: Anomalous link found with either IN or OUT node not set. %s", link)
	}

	var gene1, gene2 *Gene
	var node *network.NNode

	// Check to see if this innovation already occurred in the population
	innovationFound := false
	for _, inn := range pop.Innovations {
		/* We check to see if an innovation already occurred that was:
			-A new node
			-Stuck between the same nodes as were chosen for this mutation
			-Splitting the same gene as chosen for this mutation
		If so, we know this mutation is not a novel innovation in this generation
		so we make it match the original, identical mutation which occurred
		elsewhere in the population by coincidence */
		if inn.innovationType == newNodeInnType &&
			inn.InNodeId == inNode.Id &&
			inn.OutNodeId == outNode.Id &&
			inn.OldInnovNum == gene.InnovationNum {

			// Create the new NNode
			node = network.NewNNode(inn.NewNodeId, network.HiddenNeuron)
			// By convention, it will point to the first trait
			// Note: In future may want to change this
			node.Trait = g.Traits[0]

			// Create the new Genes
			gene1 = NewGeneWithTrait(trait, 1.0, inNode, node, link.IsRecurrent, inn.InnovationNum, 0)
			gene2 = NewGeneWithTrait(trait, oldWeight, node, outNode, false, inn.InnovationNum2, 0)

			innovationFound = true
			break
		}
	}
	// The innovation is totally novel
	if !innovationFound {
		// Get the current node id with post increment
		newNodeId := int(pop.getNextNodeIdAndIncrement())

		// Create the new NNode
		node = network.NewNNode(newNodeId, network.HiddenNeuron)
		// By convention, it will point to the first trait
		node.Trait = g.Traits[0]
		// Set node activation function as random from a list of types registered with context
		if activationType, err := context.RandomNodeActivationType(); err != nil {
			return false, err
		} else {
			node.ActivationType = activationType
		}

		// get the next innovation id for gene 1
		gene1Innovation := pop.getNextInnovationNumberAndIncrement()
		// create gene with the current gene innovation
		gene1 = NewGeneWithTrait(trait, 1.0, inNode, node, link.IsRecurrent, gene1Innovation, 0)

		// get the next innovation id for gene 2
		gene2Innovation := pop.getNextInnovationNumberAndIncrement()
		// create the second gene with this innovation incremented
		gene2 = NewGeneWithTrait(trait, oldWeight, node, outNode, false, gene2Innovation, 0)

		// Store innovation
		innovation := NewInnovationForNode(inNode.Id, outNode.Id, gene1Innovation, gene2Innovation, node.Id, gene.InnovationNum)
		pop.addInnovationSynced(innovation)
	} else if node != nil && g.hasNode(node) {
		// The same add node innovation occurred in the same genome (parent) - just skip.
		// This may happen when parent of this organism experienced the same mutation in current epoch earlier
		// and after that parent's genome was duplicated to child by mating and the same mutation parameters
		// was selected again (in_node.Id, out_node.Id, gene.InnovationNum). As result the innovation with given
		// parameters will be found and new node will be created with ID which already exists in child genome.
		// If proceed than we will have duplicated Node and genes - so we're skipping this.
		neat.InfoLog(
			fmt.Sprintf("GENOME: Add node innovation found [%t] in the same genome [%d] for node [%d]\n%s",
				innovationFound, g.Id, node.Id, g))
		return false, nil
	}

	// Now add the new NNode and new Genes to the Genome
	if node != nil && gene1 != nil && gene2 != nil {
		g.Genes = geneInsert(g.Genes, gene1)
		g.Genes = geneInsert(g.Genes, gene2)
		g.Nodes = nodeInsert(g.Nodes, node)
		return true, nil
	}
	// failed to create node or connecting genes
	return false, nil
}

// Adds Gaussian noise to link weights either GAUSSIAN or COLD_GAUSSIAN (from zero).
// The COLD_GAUSSIAN means ALL connection weights will be given completely new values
func (g *Genome) mutateLinkWeights(power, rate float64, mutationType mutatorType) (bool, error) {
	if len(g.Genes) == 0 {
		return false, errors.New("genome has no genes")
	}

	// Once in a while really shake things up
	severe := false
	if rand.Float64() > 0.5 {
		severe = true
	}

	// Go through all the Genes and perturb their link's weights
	num, genesCount := 0.0, float64(len(g.Genes))
	endPart := genesCount * 0.8
	var gaussPoint, coldGaussPoint float64

	for _, gene := range g.Genes {
		// The following if determines the probabilities of doing cold gaussian
		// mutation, meaning the probability of replacing a link weight with
		// another, entirely random weight. It is meant to bias such mutations
		// to the tail of a genome, because that is where less time-tested genes
		// reside. The gauss_point and cold_gauss_point represent values above
		// which a random float will signify that kind of mutation.
		if severe {
			gaussPoint = 0.3
			coldGaussPoint = 0.1
		} else if genesCount >= 10.0 && num > endPart {
			gaussPoint = 0.5     // Mutate by modification % of connections
			coldGaussPoint = 0.3 // Mutate the rest by replacement % of the time
		} else {
			// Half the time don't do any cold mutations
			if rand.Float64() > 0.5 {
				gaussPoint = 1.0 - rate
				coldGaussPoint = gaussPoint - 0.1
			} else {
				gaussPoint = 1.0 - rate
				coldGaussPoint = gaussPoint // no cold mutation possible (see later)
			}
		}

		random := float64(utils.RandSign()) * rand.Float64() * power
		if mutationType == gaussianMutator {
			randChoice := rand.Float64()
			if randChoice > gaussPoint {
				gene.Link.Weight += random
			} else if randChoice > coldGaussPoint {
				gene.Link.Weight = random
			}
		} else if mutationType == goldGaussianMutator {
			gene.Link.Weight = random
		}

		// Record the innovation
		gene.MutationNum = gene.Link.Weight

		num += 1.0
	}

	return true, nil
}

// Perturb params in one trait
func (g *Genome) mutateRandomTrait(context *neat.NeatContext) (bool, error) {
	if len(g.Traits) == 0 {
		return false, errors.New("genome has no traits")
	}
	// Choose a random trait number
	traitNum := rand.Intn(len(g.Traits))

	// Retrieve the trait and mutate it
	g.Traits[traitNum].Mutate(context.TraitMutationPower, context.TraitParamMutProb)

	return true, nil
}

// This chooses a random gene, extracts the link from it and re-points the link to a random trait
func (g *Genome) mutateLinkTrait(times int) (bool, error) {
	if len(g.Traits) == 0 || len(g.Genes) == 0 {
		return false, errors.New("genome has either no traits od genes")
	}
	for loop := 0; loop < times; loop++ {
		// Choose a random trait number
		traitNum := rand.Intn(len(g.Traits))

		// Choose a random link number
		geneNum := rand.Intn(len(g.Genes))

		// set the link to point to the new trait
		g.Genes[geneNum].Link.Trait = g.Traits[traitNum]

	}
	return true, nil
}

// This chooses a random node and re-points the node to a random trait specified number of times
func (g *Genome) mutateNodeTrait(times int) (bool, error) {
	if len(g.Traits) == 0 || len(g.Nodes) == 0 {
		return false, errors.New("genome has either no traits or nodes")
	}
	for loop := 0; loop < times; loop++ {
		// Choose a random trait number
		traitNum := rand.Intn(len(g.Traits))

		// Choose a random node number
		nodeNum := rand.Intn(len(g.Nodes))

		// set the node to point to the new trait
		g.Nodes[nodeNum].Trait = g.Traits[traitNum]
	}
	return true, nil
}

// Toggle genes from enable ON to enable OFF or vice versa. Do it specified number of times.
func (g *Genome) mutateToggleEnable(times int) (bool, error) {
	if len(g.Genes) == 0 {
		return false, errors.New("genome has no genes to toggle")
	}
	for loop := 0; loop < times; loop++ {
		// Choose a random gene number
		geneNum := rand.Intn(len(g.Genes))

		gene := g.Genes[geneNum]
		if gene.IsEnabled {
			// We need to make sure that another gene connects out of the in-node.
			// Because if not a section of network will break off and become isolated.
			for _, checkGene := range g.Genes {
				if checkGene.Link.InNode.Id == gene.Link.InNode.Id &&
					checkGene.IsEnabled && checkGene.InnovationNum != gene.InnovationNum {
					gene.IsEnabled = false
					break
				}
			}
		}
	}
	return true, nil
}

// Finds first disabled gene and enable it
func (g *Genome) mutateGeneReenable() (bool, error) {
	if len(g.Genes) == 0 {
		return false, errors.New("genome has no genes to re-enable")
	}
	for _, gene := range g.Genes {
		if !gene.IsEnabled {
			gene.IsEnabled = true
			break
		}
	}
	return true, nil
}

// Applies all non-structural mutations to this genome
func (g *Genome) mutateAllNonstructural(context *neat.NeatContext) (bool, error) {
	res := false
	var err error
	if rand.Float64() < context.MutateRandomTraitProb {
		// mutate random trait
		res, err = g.mutateRandomTrait(context)
	}

	if err == nil && rand.Float64() < context.MutateLinkTraitProb {
		// mutate link trait
		res, err = g.mutateLinkTrait(1)
	}

	if err == nil && rand.Float64() < context.MutateNodeTraitProb {
		// mutate node trait
		res, err = g.mutateNodeTrait(1)
	}

	if err == nil && rand.Float64() < context.MutateLinkWeightsProb {
		// mutate link weight
		res, err = g.mutateLinkWeights(context.WeightMutPower, 1.0, gaussianMutator)
	}

	if err == nil && rand.Float64() < context.MutateToggleEnableProb {
		// mutate toggle enable
		res, err = g.mutateToggleEnable(1)
	}

	if err == nil && rand.Float64() < context.MutateGeneReenableProb {
		// mutate gene reenable
		res, err = g.mutateGeneReenable()
	}
	return res, err
}

/* ****** MATING METHODS ***** */

// This method mates this Genome with another Genome g. For every point in each Genome, where each Genome shares
// the innovation number, the Gene is chosen randomly from either parent.  If one parent has an innovation absent in
// the other, the baby may inherit the innovation if it is from the more fit parent.
// The new Genome is given the id in the genomeId argument.
func (g *Genome) mateMultipoint(og *Genome, genomeId int, fitness1, fitness2 float64) (*Genome, error) {
	// Check if genomes has equal number of traits
	if len(g.Traits) != len(og.Traits) {
		return nil, errors.New(fmt.Sprintf("Genomes has different traits count, %d != %d", len(g.Traits), len(og.Traits)))
	}

	// First, average the Traits from the 2 parents to form the baby's Traits. It is assumed that trait vectors are
	// the same length. In the future, may decide on a different method for trait mating.
	newTraits, err := g.mateTraits(og)
	if err != nil {
		return nil, err
	}

	// The new genes and nodes created
	newGenes := make([]*Gene, 0)
	newNodes := make([]*network.NNode, 0)
	childNodesMap := make(map[int]*network.NNode)

	// NEW: Make sure all sensors and outputs are included (in case some inputs are disconnected)
	for _, node := range og.Nodes {
		if node.NeuronType == network.InputNeuron ||
			node.NeuronType == network.BiasNeuron ||
			node.NeuronType == network.OutputNeuron {
			nodeTraitNum := 0
			if node.Trait != nil {
				nodeTraitNum = node.Trait.Id - g.Traits[0].Id
			}
			// Create a new node off the sensor or output
			oNode := network.NewNNodeCopy(node, newTraits[nodeTraitNum])

			// Add the new node
			newNodes = nodeInsert(newNodes, oNode)
			childNodesMap[oNode.Id] = oNode
		}
	}

	// Figure out which genome is better. The worse genome should not be allowed to add extra structural baggage.
	// If they are the same, use the smaller one's disjoint and excess genes only.
	p1better := false // Tells if the first genome (this one) has better fitness or not
	if fitness1 > fitness2 ||
		(fitness1 == fitness2 && len(g.Genes) < len(og.Genes)) {
		p1better = true
	}

	// Now loop through the Genes of each parent
	i1, i2, size1, size2 := 0, 0, len(g.Genes), len(og.Genes)
	var chosenGene *Gene
	for i1 < size1 || i2 < size2 {
		skip, disable := false, false

		// choose best gene
		if i1 >= size1 {
			chosenGene = og.Genes[i2]
			i2++
			if p1better {
				skip = true // Skip excess from the worse genome
			}
		} else if i2 >= size2 {
			chosenGene = g.Genes[i1]
			i1++
			if !p1better {
				skip = true // Skip excess from the worse genome
			}
		} else {
			p1gene := g.Genes[i1]
			p2gene := og.Genes[i2]

			// Extract current innovation numbers
			p1innov := p1gene.InnovationNum
			p2innov := p2gene.InnovationNum

			if p1innov == p2innov {
				if rand.Float64() < 0.5 {
					chosenGene = p1gene
				} else {
					chosenGene = p2gene
				}

				// If one is disabled, the corresponding gene in the offspring will likely be disabled
				if !p1gene.IsEnabled || !p2gene.IsEnabled && rand.Float64() < 0.75 {
					disable = true
				}
				i1++
				i2++
			} else if p1innov < p2innov {
				chosenGene = p1gene
				i1++
				if !p1better {
					skip = true // Skip excess from the worse genome
				}
			} else {
				chosenGene = p2gene
				i2++
				if p1better {
					skip = true // Skip excess from the worse genome
				}
			}
		}

		// Uncomment this line to let growth go faster (from both parents excesses)
		// skip=false

		// Check to see if the chosen gene conflicts with an already chosen gene i.e. do they represent the same link
		for _, gene := range newGenes {
			if gene.Link.IsEqualGenetically(chosenGene.Link) {
				skip = true
				break
			}
		}

		// Now add the chosen gene to the baby
		if !skip {
			// Check for the nodes, add them if not in the baby Genome already
			inNode := chosenGene.Link.InNode
			outNode := chosenGene.Link.OutNode

			// Checking for inode's existence
			var newInNode *network.NNode
			for _, node := range newNodes {
				if node.Id == inNode.Id {
					newInNode = node
					break
				}
			}
			if newInNode == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				inNodeTraitNum := 0
				if inNode.Trait != nil {
					inNodeTraitNum = inNode.Trait.Id - g.Traits[0].Id
				}
				newInNode = network.NewNNodeCopy(inNode, newTraits[inNodeTraitNum])
				newNodes = nodeInsert(newNodes, newInNode)
				childNodesMap[newInNode.Id] = newInNode
			}

			// Checking for out node's existence
			var newOutNode *network.NNode
			for _, node := range newNodes {
				if node.Id == outNode.Id {
					newOutNode = node
					break
				}
			}
			if newOutNode == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				outNodeTraitNum := 0
				if outNode.Trait != nil {
					outNodeTraitNum = outNode.Trait.Id - g.Traits[0].Id
				}
				newOutNode = network.NewNNodeCopy(outNode, newTraits[outNodeTraitNum])
				newNodes = nodeInsert(newNodes, newOutNode)
				childNodesMap[newOutNode.Id] = newOutNode
			}

			// Add the Gene
			geneTraitNum := 0
			if chosenGene.Link.Trait != nil {
				// The subtracted number normalizes depending on whether traits start counting at 1 or 0
				geneTraitNum = chosenGene.Link.Trait.Id - g.Traits[0].Id
			}
			gene := NewGeneCopy(chosenGene, newTraits[geneTraitNum], newInNode, newOutNode)
			if disable {
				gene.IsEnabled = false
			}
			newGenes = append(newGenes, gene)
		} // end SKIP
	} // end FOR

	// check if parent's MIMO control genes should be inherited
	if len(g.ControlGenes) != 0 || len(og.ControlGenes) != 0 {
		// MIMO control genes found at least in one parent - append it to child if appropriate
		if extraNodes, modules := g.mateModules(childNodesMap, og); modules != nil {
			if len(extraNodes) > 0 {
				// append extra IO nodes of MIMO genes not found in child
				newNodes = append(newNodes, extraNodes...)
			}

			// Return modular baby genome
			return NewModularGenome(genomeId, newTraits, newNodes, newGenes, modules), nil
		}
	}
	// Return plain baby Genome
	return NewGenome(genomeId, newTraits, newNodes, newGenes), nil
}

// This method mates like multipoint but instead of selecting one or the other when the innovation numbers match,
// it averages their weights.
func (g *Genome) mateMultipointAvg(og *Genome, genomeId int, fitness1, fitness2 float64) (*Genome, error) {
	// Check if genomes has equal number of traits
	if len(g.Traits) != len(og.Traits) {
		return nil, fmt.Errorf("genomes has different traits count, %d != %d", len(g.Traits), len(og.Traits))
	}

	// First, average the Traits from the 2 parents to form the baby's Traits. It is assumed that trait vectors are
	// the same length. In the future, may decide on a different method for trait mating.
	newTraits, err := g.mateTraits(og)
	if err != nil {
		return nil, err
	}

	// The new genes and nodes created
	newGenes := make([]*Gene, 0)
	newNodes := make([]*network.NNode, 0)
	childNodesMap := make(map[int]*network.NNode)

	// NEW: Make sure all sensors and outputs are included (in case some inputs are disconnected)
	for _, node := range og.Nodes {
		if node.NeuronType == network.InputNeuron ||
			node.NeuronType == network.BiasNeuron ||
			node.NeuronType == network.OutputNeuron {
			nodeTraitNum := 0
			if node.Trait != nil {
				nodeTraitNum = node.Trait.Id - g.Traits[0].Id
			}
			// Create a new node off the sensor or output
			newNode := network.NewNNodeCopy(node, newTraits[nodeTraitNum])

			// Add the new node
			newNodes = nodeInsert(newNodes, newNode)
			childNodesMap[newNode.Id] = newNode
		}
	}

	// Figure out which genome is better. The worse genome should not be allowed to add extra structural baggage.
	// If they are the same, use the smaller one's disjoint and excess genes only.
	p1better := false // Tells if the first genome (this one) has better fitness or not
	if fitness1 > fitness2 ||
		(fitness1 == fitness2 && len(g.Genes) < len(og.Genes)) {
		p1better = true
	}

	// Set up the avgGene - this Gene is used to hold the average of the two genes to be averaged
	avgGene := NewGeneWithTrait(nil, 0.0, nil, nil, false, 0, 0.0)

	// Now loop through the Genes of each parent
	i1, i2, size1, size2 := 0, 0, len(g.Genes), len(og.Genes)
	var chosenGene *Gene
	for i1 < size1 || i2 < size2 {
		skip := false
		avgGene.IsEnabled = true // Default to enabled

		// choose best gene
		if i1 >= size1 {
			chosenGene = og.Genes[i2]
			i2++
			if p1better {
				skip = true // Skip excess from the worse genome
			}
		} else if i2 >= size2 {
			chosenGene = g.Genes[i1]
			i1++
			if !p1better {
				skip = true // Skip excess from the worse genome
			}
		} else {
			p1gene := g.Genes[i1]
			p2gene := og.Genes[i2]

			// Extract current innovation numbers
			p1innov := p1gene.InnovationNum
			p2innov := p2gene.InnovationNum

			if p1innov == p2innov {
				// Average them into the avg_gene
				if rand.Float64() > 0.5 {
					avgGene.Link.Trait = p1gene.Link.Trait
				} else {
					avgGene.Link.Trait = p2gene.Link.Trait
				}
				avgGene.Link.Weight = (p1gene.Link.Weight + p2gene.Link.Weight) / 2.0 // WEIGHTS AVERAGED HERE

				if rand.Float64() > 0.5 {
					avgGene.Link.InNode = p1gene.Link.InNode
				} else {
					avgGene.Link.InNode = p2gene.Link.InNode
				}
				if rand.Float64() > 0.5 {
					avgGene.Link.OutNode = p1gene.Link.OutNode
				} else {
					avgGene.Link.OutNode = p2gene.Link.OutNode
				}
				if rand.Float64() > 0.5 {
					avgGene.Link.IsRecurrent = p1gene.Link.IsRecurrent
				} else {
					avgGene.Link.IsRecurrent = p2gene.Link.IsRecurrent
				}

				avgGene.InnovationNum = p1innov
				avgGene.MutationNum = (p1gene.MutationNum + p2gene.MutationNum) / 2.0
				if !p1gene.IsEnabled || !p2gene.IsEnabled && rand.Float64() < 0.75 {
					avgGene.IsEnabled = false
				}

				chosenGene = avgGene
				i1++
				i2++
			} else if p1innov < p2innov {
				chosenGene = p1gene
				i1++
				if !p1better {
					skip = true // Skip excess from the worse genome
				}
			} else {
				chosenGene = p2gene
				i2++
				if p1better {
					skip = true // Skip excess from the worse genome
				}
			}
		}

		// Uncomment this line to let growth go faster (from both parents excesses)
		// skip=false

		// Check to see if the chosen gene conflicts with an already chosen gene i.e. do they represent the same link
		for _, gene := range newGenes {
			if gene.Link.IsEqualGenetically(chosenGene.Link) {
				skip = true
				break
			}
		}

		if !skip {
			// Now add the chosen gene to the baby

			// Check for the nodes, add them if not in the baby Genome already
			inNode := chosenGene.Link.InNode
			outNode := chosenGene.Link.OutNode

			// Checking for inode's existence
			var newInNode *network.NNode
			for _, node := range newNodes {
				if node.Id == inNode.Id {
					newInNode = node
					break
				}
			}
			if newInNode == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				inNodeTraitNum := 0
				if inNode.Trait != nil {
					inNodeTraitNum = inNode.Trait.Id - g.Traits[0].Id
				}
				newInNode = network.NewNNodeCopy(inNode, newTraits[inNodeTraitNum])
				newNodes = nodeInsert(newNodes, newInNode)
				childNodesMap[newInNode.Id] = newInNode
			}

			// Checking for onode's existence
			var newOutNode *network.NNode
			for _, node := range newNodes {
				if node.Id == outNode.Id {
					newOutNode = node
					break
				}
			}
			if newOutNode == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				outNodeTraitNum := 0
				if outNode.Trait != nil {
					outNodeTraitNum = outNode.Trait.Id - g.Traits[0].Id
				}
				newOutNode = network.NewNNodeCopy(outNode, newTraits[outNodeTraitNum])
				newNodes = nodeInsert(newNodes, newOutNode)
				childNodesMap[newOutNode.Id] = newOutNode
			}

			// Add the Gene
			geneTraitNum := 0
			if chosenGene.Link.Trait != nil {
				// The subtracted number normalizes depending on whether traits start counting at 1 or 0
				geneTraitNum = chosenGene.Link.Trait.Id - g.Traits[0].Id
			}
			gene := NewGeneCopy(chosenGene, newTraits[geneTraitNum], newInNode, newOutNode)
			newGenes = append(newGenes, gene)
		} // end SKIP
	} // end FOR
	// check if parent's MIMO control genes should be inherited
	if len(g.ControlGenes) != 0 || len(og.ControlGenes) != 0 {
		// MIMO control genes found at least in one parent - append it to child if appropriate
		if extraNodes, modules := g.mateModules(childNodesMap, og); modules != nil {
			if len(extraNodes) > 0 {
				// append extra IO nodes of MIMO genes not found in child
				newNodes = append(newNodes, extraNodes...)
			}

			// Return modular baby genome
			return NewModularGenome(genomeId, newTraits, newNodes, newGenes, modules), nil
		}
	}
	// Return plain baby Genome
	return NewGenome(genomeId, newTraits, newNodes, newGenes), nil
}

// This method is similar to a standard single point CROSSOVER operator. Traits are averaged as in the previous two
// mating methods. A Gene is chosen in the smaller Genome for splitting. When the Gene is reached, it is averaged with
// the matching Gene from the larger Genome, if one exists. Then every other Gene is taken from the larger Genome.
func (g *Genome) mateSinglepoint(og *Genome, genomeId int) (*Genome, error) {
	// Check if genomes has equal number of traits
	if len(g.Traits) != len(og.Traits) {
		return nil, fmt.Errorf("genomes has different traits count, %d != %d", len(g.Traits), len(og.Traits))
	}

	// First, average the Traits from the 2 parents to form the baby's Traits. It is assumed that trait vectors are
	// the same length. In the future, may decide on a different method for trait mating.
	newTraits, err := g.mateTraits(og)
	if err != nil {
		return nil, err
	}

	// The new genes and nodes created
	newGenes := make([]*Gene, 0)
	newNodes := make([]*network.NNode, 0)
	childNodesMap := make(map[int]*network.NNode)

	// NEW: Make sure all sensors and outputs are included (in case some inputs are disconnected)
	for _, node := range og.Nodes {
		if node.NeuronType == network.InputNeuron ||
			node.NeuronType == network.BiasNeuron ||
			node.NeuronType == network.OutputNeuron {
			nodeTraitNum := 0
			if node.Trait != nil {
				nodeTraitNum = node.Trait.Id - g.Traits[0].Id
			}
			// Create a new node off the sensor or output
			newNode := network.NewNNodeCopy(node, newTraits[nodeTraitNum])

			// Add the new node
			newNodes = nodeInsert(newNodes, newNode)
			childNodesMap[newNode.Id] = newNode
		}
	}

	// Set up the avg_gene - this Gene is used to hold the average of the two genes to be averaged
	avgGene := NewGeneWithTrait(nil, 0.0, nil, nil, false, 0, 0.0)

	p1stop, p2stop, stopper, crossPoint := 0, 0, 0, 0
	var p1genes, p2genes []*Gene
	size1, size2 := len(g.Genes), len(og.Genes)
	if size1 < size2 {
		crossPoint = rand.Intn(size1)
		p1stop = size1
		p2stop = size2
		stopper = size2
		p1genes = g.Genes
		p2genes = og.Genes
	} else {
		crossPoint = rand.Intn(size2)
		p1stop = size2
		p2stop = size1
		stopper = size1
		p1genes = og.Genes
		p2genes = g.Genes
	}

	var chosenGene *Gene
	geneCounter, i1, i2 := 0, 0, 0
	// Now move through the Genes of each parent until both genomes end
	for i2 < stopper {
		skip := false
		avgGene.IsEnabled = true // Default to true
		if i1 == p1stop {
			chosenGene = p2genes[i2]
			i2++
		} else if i2 == p2stop {
			chosenGene = p1genes[i1]
			i1++
		} else {
			p1gene := p1genes[i1]
			p2gene := p2genes[i2]

			// Extract current innovation numbers
			p1innov := p1gene.InnovationNum
			p2innov := p2gene.InnovationNum

			if p1innov == p2innov {
				//Pick the chosen gene depending on whether we've crossed yet
				if geneCounter < crossPoint {
					chosenGene = p1gene
				} else if geneCounter > crossPoint {
					chosenGene = p2gene
				} else {
					// We are at the crossPoint here - average genes into the avgene
					if rand.Float64() > 0.5 {
						avgGene.Link.Trait = p1gene.Link.Trait
					} else {
						avgGene.Link.Trait = p2gene.Link.Trait
					}
					avgGene.Link.Weight = (p1gene.Link.Weight + p2gene.Link.Weight) / 2.0 // WEIGHTS AVERAGED HERE

					if rand.Float64() > 0.5 {
						avgGene.Link.InNode = p1gene.Link.InNode
					} else {
						avgGene.Link.InNode = p2gene.Link.InNode
					}
					if rand.Float64() > 0.5 {
						avgGene.Link.OutNode = p1gene.Link.OutNode
					} else {
						avgGene.Link.OutNode = p2gene.Link.OutNode
					}
					if rand.Float64() > 0.5 {
						avgGene.Link.IsRecurrent = p1gene.Link.IsRecurrent
					} else {
						avgGene.Link.IsRecurrent = p2gene.Link.IsRecurrent
					}

					avgGene.InnovationNum = p1innov
					avgGene.MutationNum = (p1gene.MutationNum + p2gene.MutationNum) / 2.0
					if !p1gene.IsEnabled || !p2gene.IsEnabled && rand.Float64() < 0.75 {
						avgGene.IsEnabled = false
					}

					chosenGene = avgGene
				}
				i1++
				i2++
				geneCounter++
			} else if p1innov < p2innov {
				if geneCounter < crossPoint {
					chosenGene = p1gene
					i1++
					geneCounter++
				} else {
					chosenGene = p2gene
					i2++
				}
			} else {
				// p2innov < p1innov
				i2++
				// Special case: we need to skip to the next iteration
				// because this Gene is before the crossPoint on the wrong Genome
				skip = true
			}
		}
		if chosenGene == nil {
			// no gene was chosen - no need to process further
			skip = true
			break
		}

		// Check to see if the chosen gene conflicts with an already chosen gene i.e. do they represent the same link
		for _, gene := range newGenes {
			if gene.Link.IsEqualGenetically(chosenGene.Link) {
				skip = true
				break
			}
		}

		// Now add the chosen gene to the baby
		if !skip {
			// Check for the nodes, add them if not in the baby Genome already
			inNode := chosenGene.Link.InNode
			outNode := chosenGene.Link.OutNode

			// Checking for inode's existence
			var newInNode *network.NNode
			for _, node := range newNodes {
				if node.Id == inNode.Id {
					newInNode = node
					break
				}
			}
			if newInNode == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				inNodeTraitNum := 0
				if inNode.Trait != nil {
					inNodeTraitNum = inNode.Trait.Id - g.Traits[0].Id
				}
				newInNode = network.NewNNodeCopy(inNode, newTraits[inNodeTraitNum])
				newNodes = nodeInsert(newNodes, newInNode)
				childNodesMap[newInNode.Id] = newInNode
			}

			// Checking for onode's existence
			var newOutNode *network.NNode
			for _, node := range newNodes {
				if node.Id == outNode.Id {
					newOutNode = node
					break
				}
			}
			if newOutNode == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				outNodeTraitNum := 0
				if outNode.Trait != nil {
					outNodeTraitNum = outNode.Trait.Id - g.Traits[0].Id
				}
				newOutNode = network.NewNNodeCopy(outNode, newTraits[outNodeTraitNum])
				newNodes = nodeInsert(newNodes, newOutNode)
				childNodesMap[newOutNode.Id] = newOutNode
			}

			// Add the Gene
			geneTraitNum := 0
			if chosenGene.Link.Trait != nil {
				// The subtracted number normalizes depending on whether traits start counting at 1 or 0
				geneTraitNum = chosenGene.Link.Trait.Id - g.Traits[0].Id
			}
			gene := NewGeneCopy(chosenGene, newTraits[geneTraitNum], newInNode, newOutNode)
			newGenes = append(newGenes, gene)
		} // end SKIP
	} // end FOR
	// check if parent's MIMO control genes should be inherited
	if len(g.ControlGenes) != 0 || len(og.ControlGenes) != 0 {
		// MIMO control genes found at least in one parent - append it to child if appropriate
		if extraNodes, modules := g.mateModules(childNodesMap, og); modules != nil {
			if len(extraNodes) > 0 {
				// append extra IO nodes of MIMO genes not found in child
				newNodes = append(newNodes, extraNodes...)
			}

			// Return modular baby genome
			return NewModularGenome(genomeId, newTraits, newNodes, newGenes, modules), nil
		}
	}
	// Return plain baby Genome
	return NewGenome(genomeId, newTraits, newNodes, newGenes), nil
}

// Builds an array of modules to be added to the child during crossover.
// If any or both parents has module and at least one modular endpoint node already inherited by child genome than make
// sure that child get all associated module nodes
func (g *Genome) mateModules(childNodes map[int]*network.NNode, og *Genome) ([]*network.NNode, []*MIMOControlGene) {
	parentModules := make([]*MIMOControlGene, 0)
	currGenomeModules := findModulesIntersection(childNodes, g.ControlGenes)
	if len(currGenomeModules) > 0 {
		parentModules = append(parentModules, currGenomeModules...)
	}
	outGenomeModules := findModulesIntersection(childNodes, og.ControlGenes)
	if len(outGenomeModules) > 0 {
		parentModules = append(parentModules, outGenomeModules...)
	}
	if len(parentModules) == 0 {
		return nil, nil
	}

	// collect IO nodes from all included modules and add return it as extra ones
	extraNodes := make([]*network.NNode, 0)
	for _, cg := range parentModules {
		for _, n := range cg.ioNodes {
			if _, ok := childNodes[n.Id]; !ok {
				// not found in known child nodes - collect it
				extraNodes = append(extraNodes, n)
			}
		}
	}

	return extraNodes, parentModules
}

// Finds intersection of provided nodes with IO nodes from control genes and returns list of control genes found.
// If no intersection found empty list returned.
func findModulesIntersection(nodes map[int]*network.NNode, genes []*MIMOControlGene) []*MIMOControlGene {
	modules := make([]*MIMOControlGene, 0)
	for _, cg := range genes {
		if cg.hasIntersection(nodes) {
			modules = append(modules, cg)
		}
	}
	return modules
}

// Builds array of traits for child genome during crossover
func (g *Genome) mateTraits(og *Genome) ([]*neat.Trait, error) {
	newTraits := make([]*neat.Trait, len(g.Traits))
	var err error
	for i, tr := range g.Traits {
		newTraits[i], err = neat.NewTraitAvrg(tr, og.Traits[i]) // construct by averaging
		if err != nil {
			return nil, err
		}
	}
	return newTraits, nil
}

/* ******** COMPATIBILITY CHECKING METHODS * ********/

// This function gives a measure of compatibility between two Genomes by computing a linear combination of three
// characterizing variables of their compatibility. The three variables represent PERCENT DISJOINT GENES,
// PERCENT EXCESS GENES, MUTATIONAL DIFFERENCE WITHIN MATCHING GENES. So the formula for compatibility
// is:  disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg
// The three coefficients are global system parameters.
// The bigger returned value the less compatible the genomes. Fully compatible genomes has 0.0 returned.
func (g *Genome) compatibility(og *Genome, context *neat.NeatContext) float64 {
	if context.GenCompatMethod == 0 {
		return g.compatLinear(og, context)
	} else {
		return g.compatFast(og, context)
	}
}

// The compatibility checking method with linear performance depending on the size of the lognest genome in comparison.
// When genomes are small this method is compatible in performance with Genome#compatFast method.
// The compatibility formula remains the same: disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg
// where: pdg - PERCENT DISJOINT GENES, peg - PERCENT EXCESS GENES, and mdmg - MUTATIONAL DIFFERENCE WITHIN MATCHING GENES
func (g *Genome) compatLinear(og *Genome, context *neat.NeatContext) float64 {
	numDisjoint, numExcess, mutDiffTotal, numMatching := 0.0, 0.0, 0.0, 0.0
	size1, size2 := len(g.Genes), len(og.Genes)
	maxGenomeSize := size2
	if size1 > size2 {
		maxGenomeSize = size1
	}
	var gene1, gene2 *Gene
	for i, i1, i2 := 0, 0, 0; i < maxGenomeSize; i++ {
		if i1 >= size1 {
			numExcess += 1.0
			i2++
		} else if i2 >= size2 {
			numExcess += 1.0
			i1++
		} else {
			gene1 = g.Genes[i1]
			gene2 = og.Genes[i2]
			p1innov := gene1.InnovationNum
			p2innov := gene2.InnovationNum

			if p1innov == p2innov {
				numMatching += 1.0
				mutDiff := math.Abs(gene1.MutationNum - gene2.MutationNum)
				mutDiffTotal += mutDiff
				i1++
				i2++
			} else if p1innov < p2innov {
				i1++
				numDisjoint += 1.0
			} else if p2innov < p1innov {
				i2++
				numDisjoint += 1.0
			}
		}
	}

	//fmt.Printf("num_disjoint: %.f num_excess: %.f mut_diff_total: %.f num_matching: %.f\n", num_disjoint, num_excess, mut_diff_total, num_matching)

	// Return the compatibility number using compatibility formula
	// Note that mut_diff_total/num_matching gives the AVERAGE difference between mutation_nums for any two matching
	// Genes in the Genome. Look at disjointedness and excess in the absolute (ignoring size)
	comp := context.DisjointCoeff*numDisjoint + context.ExcessCoeff*numExcess +
		context.MutdiffCoeff*(mutDiffTotal/numMatching)

	return comp
}

// The faster version of genome compatibility checking. The compatibility check will start from the end of genome where
// the most of disparities are located - the novel genes with greater innovation ID are always attached at the end (see geneInsert).
// This has the result of complicating the routine because we must now invoke additional logic to determine which genes
// are excess and when the first disjoint gene is found. This is done with an extra integer:
// * excessGenesSwitch=0 // indicates to the loop that it is handling the first gene.
// * excessGenesSwitch=1 // Indicates that the first gene was excess and on genome 1.
// * excessGenesSwitch=2 // Indicates that the first gene was excess and on genome 2.
// * excessGenesSwitch=3 // Indicates that there are no more excess genes.
//
// The compatibility formula remains the same: disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg
// where: pdg - PERCENT DISJOINT GENES, peg - PERCENT EXCESS GENES, and mdmg - MUTATIONAL DIFFERENCE WITHIN MATCHING GENES
func (g *Genome) compatFast(og *Genome, context *neat.NeatContext) float64 {
	list1Count, list2Count := len(g.Genes), len(og.Genes)
	// First test edge cases
	if list1Count == 0 && list2Count == 0 {
		// Both lists are empty! No disparities, therefore the genomes are compatible!
		return 0.0
	}
	if list1Count == 0 {
		// All list2 genes are excess.
		return float64(list2Count) * context.ExcessCoeff
	}

	if list2Count == 0 {
		// All list1 genes are excess.
		return float64(list1Count) * context.ExcessCoeff
	}

	excessGenesSwitch, numMatching := 0, 0
	compatibility, mutDiff := 0.0, 0.0
	list1Idx, list2Idx := list1Count-1, list2Count-1
	gene1, gene2 := g.Genes[list1Idx], og.Genes[list2Idx]

	for {
		if gene2.InnovationNum > gene1.InnovationNum {
			// Most common test case(s) at top for efficiency.
			if excessGenesSwitch == 3 {
				// No more excess genes. Therefore this mismatch is disjoint.
				compatibility += context.DisjointCoeff
			} else if excessGenesSwitch == 2 {
				// Another excess gene on genome 2.
				compatibility += context.ExcessCoeff
			} else if excessGenesSwitch == 1 {
				// We have found the first non-excess gene.
				excessGenesSwitch = 3
				compatibility += context.DisjointCoeff
			} else {
				// First gene is excess, and is on genome 2.
				excessGenesSwitch = 2
				compatibility += context.ExcessCoeff
			}

			// Move to the next gene in list2.
			list2Idx--
		} else if gene1.InnovationNum == gene2.InnovationNum {
			// No more excess genes. It's quicker to set this every time than to test if is not yet 3.
			excessGenesSwitch = 3

			// Matching genes. Increase compatibility by MutationNum difference * coeff.
			mutDiff += math.Abs(gene1.MutationNum - gene2.MutationNum)
			numMatching++

			// Move to the next gene in both lists.
			list1Idx--
			list2Idx--
		} else {
			// Most common test case(s) at top for efficiency.
			if excessGenesSwitch == 3 {
				// No more excess genes. Therefore this mismatch is disjoint.
				compatibility += context.DisjointCoeff
			} else if excessGenesSwitch == 1 {
				// Another excess gene on genome 1.
				compatibility += context.ExcessCoeff
			} else if excessGenesSwitch == 2 {
				// We have found the first non-excess gene.
				excessGenesSwitch = 3
				compatibility += context.DisjointCoeff
			} else {
				// First gene is excess, and is on genome 1.
				excessGenesSwitch = 1
				compatibility += context.ExcessCoeff
			}

			// Move to the next gene in list1.
			list1Idx--
		}

		// Check if we have reached the end of one (or both) of the lists. If we have reached the end of both then
		// we execute the first 'if' block - but it doesn't matter since the loop is not entered if both lists have
		// been exhausted.
		if list1Idx < 0 {
			// All remaining list2 genes are disjoint.
			compatibility += float64(list2Idx+1) * context.DisjointCoeff
			break

		}

		if list2Idx < 0 {
			// All remaining list1 genes are disjoint.
			compatibility += float64(list1Idx+1) * context.DisjointCoeff
			break
		}

		gene1, gene2 = g.Genes[list1Idx], og.Genes[list2Idx]
	}
	if numMatching > 0 {
		compatibility += mutDiff * context.MutdiffCoeff / float64(numMatching)
	}
	return compatibility
}
