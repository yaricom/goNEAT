package genetics

import (
	"errors"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"io"
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
	Id int `yaml:"id"`
	// The parameters conglomerations
	Traits []*neat.Trait `yaml:"traits"`
	// List of NNodes for the Network
	Nodes []*network.NNode `yaml:"nodes"`
	// List of innovation-tracking genes
	Genes []*Gene `yaml:"genes"`
	// List of MIMO control genes
	ControlGenes []*MIMOControlGene `yaml:"modules"`

	// Allows Genome to be matched with its Network
	Phenotype *network.Network `yaml:""`
}

// NewGenome Constructor which takes full genome specs and puts them into the new one
func NewGenome(id int, t []*neat.Trait, n []*network.NNode, g []*Gene) *Genome {
	return &Genome{
		Id:     id,
		Traits: t,
		Nodes:  n,
		Genes:  g,
	}
}

// NewModularGenome Constructs new modular genome
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
					weight := float64(math.RandSign()) * rand.Float64()
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

// ReadGenome reads Genome from reader
func ReadGenome(ir io.Reader, id int) (*Genome, error) {
	// stub for backward compatibility
	// the new implementations should use GenomeReader to decode genome data in variety of formats
	r, err := NewGenomeReader(ir, PlainGenomeEncoding)
	if err != nil {
		return nil, err
	}
	if gnome, err := r.Read(); err != nil {
		return nil, err
	} else {
		gnome.Id = id
		return gnome, err
	}
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

// Extrons Return # of non-disabled genes
func (g *Genome) Extrons() int {
	total := 0
	for _, gene := range g.Genes {
		if gene.IsEnabled {
			total++
		}
	}
	return total
}

// IsEqual Tests if given genome is equal to this one genetically and phenotypically. This method will check that both
// genomes has the same traits, nodes and genes.
// If mismatch detected the error will be returned with mismatch details.
func (g *Genome) IsEqual(og *Genome) (bool, error) {
	if len(g.Traits) != len(og.Traits) {
		return false, fmt.Errorf("traits count mismatch: %d != %d",
			len(g.Traits), len(og.Traits))
	}
	for i, tr := range og.Traits {
		if !reflect.DeepEqual(tr, g.Traits[i]) {
			return false, fmt.Errorf("traits mismatch, expected: %s, but found: %s", tr, g.Traits[i])
		}
	}

	if len(g.Nodes) != len(og.Nodes) {
		return false, fmt.Errorf("nodes count mismatch: %d != %d",
			len(g.Nodes), len(og.Nodes))
	}
	for i, nd := range og.Nodes {
		if !reflect.DeepEqual(nd, g.Nodes[i]) {
			return false, fmt.Errorf("node mismatch, expected: %s\nfound: %s", nd, g.Nodes[i])
		}
	}

	if len(g.Genes) != len(og.Genes) {
		return false, fmt.Errorf("genes count mismatch: %d != %d",
			len(g.Genes), len(og.Genes))
	}
	for i, gen := range og.Genes {
		if !reflect.DeepEqual(gen, g.Genes[i]) {
			return false, fmt.Errorf("gene mismatch, expected: %s\nfound: %s", gen, g.Genes[i])
		}
	}

	if len(g.ControlGenes) != len(og.ControlGenes) {
		return false, fmt.Errorf("control genes count mismatch: %d != %d",
			len(g.ControlGenes), len(og.ControlGenes))
	}
	for i, cg := range og.ControlGenes {
		if !reflect.DeepEqual(cg, g.ControlGenes[i]) {
			return false, fmt.Errorf("control gene mismatch, expected: %s\nfound: %s", cg, g.ControlGenes[i])
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
		// The gene has innovation number higher that not assigned yet innovation number for this genome. This means
		// that this is gene not from this genome lineage.
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

// Genesis generates a Network phenotype from this Genome with specified id
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
			newLink = network.NewLinkWithTrait(curLink.Trait, curLink.ConnectionWeight, inNode, outNode, curLink.IsRecurrent)

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
					newLink = network.NewLink(l.ConnectionWeight, inNode, outNode, false)
					// only incoming to control node
					outNode.Incoming = append(outNode.Incoming, newLink)
				}

				// connect outputs
				for _, l := range cg.ControlNode.Outgoing {
					inNode = newCopyNode
					outNode = l.OutNode.PhenotypeAnalogue
					newLink = network.NewLink(l.ConnectionWeight, inNode, outNode, false)
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
	traitsDup := make([]*neat.Trait, len(g.Traits))
	for i, tr := range g.Traits {
		traitsDup[i] = neat.NewTraitCopy(tr)
	}

	// Duplicate NNodes
	nodesDup := make([]*network.NNode, len(g.Nodes))
	for i, nd := range g.Nodes {
		// First, find the duplicate of the trait that this node points to
		assocTrait := nd.Trait
		if assocTrait != nil {
			assocTrait = TraitWithId(assocTrait.Id, traitsDup)
		}
		nodesDup[i] = network.NewNNodeCopy(nd, assocTrait)
	}

	// Duplicate Genes
	genesDup := make([]*Gene, len(g.Genes))
	for i, gn := range g.Genes {
		// First find the nodes connected by the gene's link
		inNode := NodeWithId(gn.Link.InNode.Id, nodesDup)
		if inNode == nil {
			return nil, fmt.Errorf("incoming node: %d not found for gene %s",
				gn.Link.InNode.Id, gn.String())
		}
		outNode := NodeWithId(gn.Link.OutNode.Id, nodesDup)
		if outNode == nil {
			return nil, fmt.Errorf("outgoing node: %d not found for gene %s",
				gn.Link.OutNode.Id, gn.String())
		}

		// Find the duplicate of trait associated with this gene
		assocTrait := gn.Link.Trait
		if assocTrait != nil {
			assocTrait = TraitWithId(assocTrait.Id, traitsDup)
		}

		genesDup[i] = NewGeneCopy(gn, assocTrait, inNode, outNode)
	}

	if len(g.ControlGenes) == 0 {
		// If no MIMO control genes return plain genome
		return NewGenome(newId, traitsDup, nodesDup, genesDup), nil
	} else {
		// Duplicate MIMO Control Genes and build modular genome
		controlGenesDup := make([]*MIMOControlGene, len(g.ControlGenes))
		for i, cg := range g.ControlGenes {
			// duplicate control node
			controlNode := cg.ControlNode
			// find duplicate of trait associated with control node
			assocTrait := controlNode.Trait
			if assocTrait != nil {
				assocTrait = TraitWithId(assocTrait.Id, traitsDup)
			}
			nodeCopy := network.NewNNodeCopy(controlNode, assocTrait)
			// add incoming links
			for _, l := range controlNode.Incoming {
				inNode := NodeWithId(l.InNode.Id, nodesDup)
				if inNode == nil {
					return nil, fmt.Errorf("incoming node: %d not found for control node: %d",
						l.InNode.Id, controlNode.Id)
				}
				newInLink := network.NewLinkCopy(l, inNode, nodeCopy)
				nodeCopy.Incoming = append(nodeCopy.Incoming, newInLink)
			}

			// add outgoing links
			for _, l := range controlNode.Outgoing {
				outNode := NodeWithId(l.OutNode.Id, nodesDup)
				if outNode == nil {
					return nil, fmt.Errorf("outgoing node: %d not found for control node: %d",
						l.InNode.Id, controlNode.Id)
				}
				newOutLink := network.NewLinkCopy(l, nodeCopy, outNode)
				nodeCopy.Outgoing = append(nodeCopy.Outgoing, newOutLink)
			}

			// add MIMO control gene
			controlGenesDup[i] = NewMIMOGeneCopy(cg, nodeCopy)
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
			return false, errors.New("missing input node of gene in the genome nodes list")
		}
		if !outFound {
			return false, errors.New("missing output node of gene in the genome nodes list")
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
				return false, fmt.Errorf("duplicate genes found: %s == %s", gn, gn2)
			}
		}
	}
	// Check for 2 disables in a row
	// Note: Again, this is not necessarily a bad sign
	if len(g.Nodes) > 500 {
		disabled := false
		for _, gn := range g.Genes {
			if !gn.IsEnabled && disabled {
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
