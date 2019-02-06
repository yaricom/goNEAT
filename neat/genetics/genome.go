package genetics

import (
	"github.com/yaricom/goNEAT/neat/utils"
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat"
	"math/rand"
	"io"
	"fmt"
	"errors"
	"math"
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
	Id           int
	// The parameters conglomerations
	Traits       []*neat.Trait
	// List of NNodes for the Network
	Nodes        []*network.NNode
	// List of innovation-tracking genes
	Genes        []*Gene
	// List of MIMO control genes
	ControlGenes []*MIMOControlGene

	// Allows Genome to be matched with its Network
	Phenotype    *network.Network
}

// Constructor which takes full genome specs and puts them into the new one
func NewGenome(id int, t []*neat.Trait, n []*network.NNode, g []*Gene) *Genome {
	return &Genome{
		Id:id,
		Traits:t,
		Nodes:n,
		Genes:g,
	}
}

// Constructs new modular genome
func NewModularGenome(id int, t []*neat.Trait, n []*network.NNode, g []*Gene, mimoG []*MIMOControlGene) *Genome {
	return &Genome{
		Id:id,
		Traits:t,
		Nodes:n,
		Genes:g,
		ControlGenes:mimoG,
	}
}

// This special constructor creates a Genome with in inputs, out outputs, n out of nmax hidden units, and random
// connectivity.  If rec is true then recurrent connections will be included. The last input is a bias
// link_prob is the probability of a link. The created genome is not modular.
func newGenomeRand(new_id, in, out, n, nmax int, recurrent bool, link_prob float64) *Genome {
	total_nodes := in + out + nmax
	matrix_dim := total_nodes * total_nodes
	// The connection matrix which will be randomized
	cm := make([]bool, matrix_dim)  //Dimension the connection matrix

	// No nodes above this number for this genome
	max_node := in + n
	first_output := total_nodes - out + 1

	// For creating the new genes
	var new_node, in_node, out_node *network.NNode
	var new_gene *Gene
	var new_trait *neat.Trait

	// Create a dummy trait (this is for future expansion of the system)
	new_trait = neat.NewTrait()
	new_trait.Id = 1
	new_trait.Params = make([]float64, neat.Num_trait_params)

	// Create empty genome
	gnome := Genome{
		Id:new_id,
		Traits:[]*neat.Trait{new_trait},
		Nodes:make([]*network.NNode, 0),
		Genes:make([]*Gene, 0),
	}

	// Step through the connection matrix, randomly assigning bits
	for count := 0; count < matrix_dim; count++ {
		cm[count] = (rand.Float64() < link_prob)
	}

	// Build the input nodes
	for ncount := 1; ncount <= in; ncount++ {
		if ncount < in {
			new_node = network.NewNNode(ncount, network.InputNeuron)
		} else {
			new_node = network.NewNNode(ncount, network.BiasNeuron)
		}
		new_node.Trait = new_trait
		gnome.Nodes = append(gnome.Nodes, new_node)
	}

	// Build the hidden nodes
	for ncount := in + 1; ncount <= in + n; ncount++ {
		new_node = network.NewNNode(ncount, network.HiddenNeuron)
		new_node.Trait = new_trait
		gnome.Nodes = append(gnome.Nodes, new_node)
	}

	// Build the output nodes
	for ncount := first_output; ncount <= total_nodes; ncount++ {
		new_node = network.NewNNode(ncount, network.OutputNeuron)
		new_node.Trait = new_trait
		gnome.Nodes = append(gnome.Nodes, new_node)
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

	// Step through the connection matrix, creating connection genes
	count := 0
	var create_gene, flag_recurrent bool
	for col := 1; col <= total_nodes; col++ {
		for row := 1; row <= total_nodes; row++ {
			// Only try to create a link if it is in the matrix and not leading into a sensor
			if cm[count] && col > in &&
				(col <= max_node || col >= first_output) &&
				(row <= max_node || row >= first_output) {

				// If it's recurrent, create the connection (gene) no matter what
				create_gene = true
				if col > row {
					flag_recurrent = false
				} else {
					flag_recurrent = true
					if !recurrent {
						// skip recurrent connections
						create_gene = false
					}
				}

				// Introduce new connection (gene) into genome
				if create_gene {
					// Retrieve in_node and out_node
					for i := 0; i < len(gnome.Nodes) && (in_node == nil || out_node == nil); i++ {
						node_id := gnome.Nodes[i].Id
						if node_id == row {
							in_node = gnome.Nodes[i]
						}
						if node_id == col {
							out_node = gnome.Nodes[i]
						}
					}

					// Create the gene
					new_weight := float64(utils.RandSign()) * rand.Float64()
					new_gene = NewGeneWithTrait(new_trait, new_weight, in_node, out_node, flag_recurrent, int64(count), new_weight)

					//Add the gene to the genome
					gnome.Genes = append(gnome.Genes, new_gene)
				}

			}

			count++ //increment counter
			// reset nodes
			in_node, out_node = nil, nil
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
		n_type := ""
		switch n.NeuronType {
		case network.InputNeuron:
			n_type = "I"
		case network.OutputNeuron:
			n_type = "O"
		case network.BiasNeuron:
			n_type = "B"
		case network.HiddenNeuron:
			n_type = "H"
		}
		str += fmt.Sprintf("\t%s%s \n", n_type, n)
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
		return -1, errors.New("Genome has no nodes")
	}
	id := g.Nodes[len(g.Nodes) - 1].Id
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
	inn_num := int64(0)
	// check connection genes
	if len(g.Genes) > 0 {
		inn_num = g.Genes[len(g.Genes) - 1].InnovationNum
	} else {
		return -1, errors.New("Genome has no Genes")
	}
	// check control genes if any
	if len(g.ControlGenes) > 0 {
		c_inn_num := g.ControlGenes[len(g.ControlGenes) - 1].InnovationNum
		if c_inn_num > inn_num {
			inn_num = c_inn_num
		}
	}
	return inn_num + int64(1), nil
}

// Returns true if this Genome already includes provided node
func (g *Genome) hasNode(node *network.NNode) bool {
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
func (g *Genome) Genesis(net_id int) (*network.Network, error) {
	// Inputs and outputs will be collected here for the network.
	// All nodes are collected in an all_list -
	// this is useful for network traversing routines
	in_list := make([]*network.NNode, 0)
	out_list := make([]*network.NNode, 0)
	all_list := make([]*network.NNode, 0)

	var new_node *network.NNode
	// Create the network nodes
	for _, n := range g.Nodes {
		new_node = network.NewNNodeCopy(n, n.Trait)

		// Check for input or output designation of node
		if n.NeuronType == network.InputNeuron || n.NeuronType == network.BiasNeuron {
			in_list = append(in_list, new_node)
		} else if n.NeuronType == network.OutputNeuron {
			out_list = append(out_list, new_node)
		}

		// Keep track of all nodes in one place for convenience
		all_list = append(all_list, new_node)

		// Have the node specifier point to the node it generated
		n.PhenotypeAnalogue = new_node
	}

	if len(g.Genes) == 0 {
		return nil, errors.New("The network built whitout GENES; the result can be unpredictable")
	}

	if len(out_list) == 0 {
		return nil, errors.New(fmt.Sprintf("The network whitout OUTPUTS; the result can be unpredictable. Genome: %s", g))
	}

	var in_node, out_node *network.NNode
	var cur_link, new_link *network.Link
	// Create the links by iterating through the genes
	for _, gn := range g.Genes {
		// Only create the link if the gene is enabled
		if gn.IsEnabled {
			cur_link = gn.Link
			in_node = cur_link.InNode.PhenotypeAnalogue
			out_node = cur_link.OutNode.PhenotypeAnalogue

			// NOTE: This line could be run through a recurrency check if desired
			// (no need to in the current implementation of NEAT)
			new_link = network.NewLinkWithTrait(cur_link.Trait, cur_link.Weight, in_node, out_node, cur_link.IsRecurrent)

			// Add link to the connected nodes
			out_node.Incoming = append(out_node.Incoming, new_link)
			in_node.Outgoing = append(in_node.Outgoing, new_link)
		}
	}

	var new_net *network.Network
	if len(g.ControlGenes) == 0 {
		// Create the new network
		new_net = network.NewNetwork(in_list, out_list, all_list, net_id)
	} else {
		// Create MIMO control genes
		c_nodes := make([]*network.NNode, 0)
		for _, cg := range g.ControlGenes {
			// Only process enabled genes
			if cg.IsEnabled {
				new_c_node := network.NewNNodeCopy(cg.ControlNode, cg.ControlNode.Trait)

				// connect inputs
				for _, l := range cg.ControlNode.Incoming {
					in_node = l.InNode.PhenotypeAnalogue
					out_node = new_c_node
					new_link = network.NewLink(l.Weight, in_node, out_node, false)
					// only incoming to control node
					out_node.Incoming = append(out_node.Incoming, new_link)
				}

				// connect outputs
				for _, l := range cg.ControlNode.Outgoing {
					in_node = new_c_node
					out_node = l.OutNode.PhenotypeAnalogue
					new_link = network.NewLink(l.Weight, in_node, out_node, false)
					// only outgoing from control node
					in_node.Outgoing = append(in_node.Outgoing, new_link)
				}

				// store control node
				c_nodes = append(c_nodes, new_c_node)
			}
		}
		new_net = network.NewModularNetwork(in_list, out_list, all_list, c_nodes, net_id)
	}

	// Attach genotype and phenotype together:
	// genotype points to owner phenotype (new_net)
	g.Phenotype = new_net

	return new_net, nil
}

// Duplicate this Genome to create a new one with the specified id
func (g *Genome) duplicate(new_id int) (*Genome, error) {

	// Duplicate the traits
	traits_dup := make([]*neat.Trait, 0)
	for _, tr := range g.Traits {
		new_trait := neat.NewTraitCopy(tr)
		traits_dup = append(traits_dup, new_trait)
	}

	// Duplicate NNodes
	nodes_dup := make([]*network.NNode, 0)
	for _, nd := range g.Nodes {
		// First, find the duplicate of the trait that this node points to
		assoc_trait := nd.Trait
		if assoc_trait != nil {
			assoc_trait = traitWithId(assoc_trait.Id, traits_dup)
		}
		new_node := network.NewNNodeCopy(nd, assoc_trait)

		nodes_dup = append(nodes_dup, new_node)
	}

	// Duplicate Genes
	genes_dup := make([]*Gene, 0)
	for _, gn := range g.Genes {
		// First find the nodes connected by the gene's link
		in_node := nodeWithId(gn.Link.InNode.Id, nodes_dup)
		if in_node == nil {
			return nil, errors.New(
				fmt.Sprintf("incoming node: %d not found for gene %s",
					gn.Link.InNode.Id, gn.String()))
		}
		out_node := nodeWithId(gn.Link.OutNode.Id, nodes_dup)
		if out_node == nil {
			return nil, errors.New(
				fmt.Sprintf("outgoing node: %d not found for gene %s",
					gn.Link.OutNode.Id, gn.String()))
		}

		// Find the duplicate of trait associated with this gene
		assoc_trait := gn.Link.Trait
		if assoc_trait != nil {
			assoc_trait = traitWithId(assoc_trait.Id, traits_dup)
		}

		new_gene := NewGeneCopy(gn, assoc_trait, in_node, out_node)
		genes_dup = append(genes_dup, new_gene)
	}

	if len(g.ControlGenes) == 0 {
		// If no MIMO control genes return plain genome
		return NewGenome(new_id, traits_dup, nodes_dup, genes_dup), nil
	} else {
		// Duplicate MIMO Control Genes and build modular genome
		control_genes_dup := make([]*MIMOControlGene, 0)
		for _, cg := range g.ControlGenes {
			// duplicate control node
			c_node := cg.ControlNode
			// find duplicate of trait associated with control node
			assoc_trait := c_node.Trait
			if assoc_trait != nil {
				assoc_trait = traitWithId(assoc_trait.Id, traits_dup)
			}
			new_c_node := network.NewNNodeCopy(c_node, assoc_trait)
			// add incoming links
			for _, l := range c_node.Incoming {
				in_node := nodeWithId(l.InNode.Id, nodes_dup)
				if in_node == nil {
					return nil, errors.New(
						fmt.Sprintf("incoming node: %d not found for control node: %d",
							l.InNode.Id, c_node.Id))
				}
				new_in_link := network.NewLinkCopy(l, in_node, new_c_node)
				new_c_node.Incoming = append(new_c_node.Incoming, new_in_link)
			}

			// add outgoing links
			for _, l := range c_node.Outgoing {
				out_node := nodeWithId(l.OutNode.Id, nodes_dup)
				if out_node == nil {
					return nil, errors.New(
						fmt.Sprintf("outgoing node: %d not found for control node: %d",
							l.InNode.Id, c_node.Id))
				}
				new_out_link := network.NewLinkCopy(l, new_c_node, out_node)
				new_c_node.Outgoing = append(new_c_node.Outgoing, new_out_link)
			}

			// add MIMO control gene
			new_cg := NewMIMOGeneCopy(cg, new_c_node)
			control_genes_dup = append(control_genes_dup, new_cg)
		}

		return NewModularGenome(new_id, traits_dup, nodes_dup, genes_dup, control_genes_dup), nil
	}
}

// For debugging: A number of tests can be run on a genome to check its integrity.
// Note: Some of these tests do not indicate a bug, but rather are meant to be used to detect specific system states.
func (g *Genome) verify() (bool, error) {
	if len(g.Genes) == 0 {
		return false, errors.New("Genome has no Genes")
	}
	if len(g.Nodes) == 0 {
		return false, errors.New("Genome has no Nodes")
	}
	if len(g.Traits) == 0 {
		return false, errors.New("Genome has no Traits")
	}


	// Check each gene's nodes
	for _, gn := range g.Genes {
		inode := gn.Link.InNode
		onode := gn.Link.OutNode
		i_found, o_found := false, false
		for i := 0; i < len(g.Nodes) && (!i_found || !o_found); i++ {
			if inode.Id == g.Nodes[i].Id {
				i_found = true
			}
			if onode.Id == g.Nodes[i].Id {
				o_found = true
			}
		}

		// check results
		if !i_found {
			return false, errors.New("Missing input node of gene in the genome nodes")
		}
		if !o_found {
			return false, errors.New("Missing output node of gene in the genome nodes")
		}
	}

	// Check for NNodes being out of order
	last_id := 0
	for _, n := range g.Nodes {
		if n.Id < last_id {
			return false, errors.New("Nodes out of order in genome")
		}
		last_id = n.Id
	}

	// Make sure there are no duplicate genes
	for _, gn := range g.Genes {
		for _, gn2 := range g.Genes {
			if gn != gn2 && gn.Link.IsEqualGenetically(gn2.Link) {
				return false, errors.New(fmt.Sprintf("Duplicate genes found. %s == %s", gn, gn2))
			}
		}
	}
	// Check for 2 disables in a row
	// Note: Again, this is not necessarily a bad sign
	if len(g.Nodes) > 500 {
		disab := false
		for _, gn := range g.Genes {
			if gn.IsEnabled == false && disab {
				return false, errors.New("Two gene disables in a row")
			}
			disab = !gn.IsEnabled
		}
	}
	return true, nil
}

// Inserts a NNode into a given ordered list of NNodes in ascending order by NNode ID
func nodeInsert(nodes[]*network.NNode, n *network.NNode) []*network.NNode {
	index := len(nodes)
	// quick insert at the end or beginning (we assume that nodes is already ordered)
	if index == 0 || n.Id >= nodes[index - 1].Id {
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
	first := make([]*network.NNode, index + 1)
	copy(first, nodes[0:index])
	first[index] = n
	second := nodes[index:]

	nodes = append(first, second...)
	return nodes
}

// Inserts a new gene that has been created through a mutation in the
// *correct order* into the list of genes in the genome, i.e. ordered by innovation number ascending
func geneInsert(genes[]*Gene, g *Gene) []*Gene {
	index := len(genes) // to make sure that greater IDs appended at the end
	// quick insert at the end or beginning (we assume that nodes is already ordered)
	if index == 0 || g.InnovationNum >= genes[index - 1].InnovationNum {
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

	first := make([]*Gene, index + 1)
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
func (g *Genome) mutateConnectSensors(pop *Population, context *neat.NeatContext) (bool, error) {

	if len(g.Genes) == 0 {
		return false, errors.New("Genome has no genes")
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
	disconnected_sensors := make([]*network.NNode, 0)
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
			disconnected_sensors = append(disconnected_sensors, sensor)
		}
	}

	// if all sensors are connected - stop
	if len(disconnected_sensors) == 0 {
		return false, nil
	}

	// pick randomly from disconnected sensors
	sensor := disconnected_sensors[rand.Intn(len(disconnected_sensors))]
	// add new links to chosen sensor, avoiding redundancy
	link_added := false
	for _, output := range outputs {
		found := false
		for _, gene := range g.Genes {
			if gene.Link.InNode == sensor && gene.Link.OutNode == output {
				found = true
				break
			}
		}

		if !found {
			var new_gene *Gene
			// Check to see if this innovation already occurred in the population
			innovation_found := false
			for _, inn := range pop.Innovations {
				if inn.innovationType == newLinkInnType &&
					inn.InNodeId == sensor.Id &&
					inn.OutNodeId == output.Id &&
					inn.IsRecurrent == false {

					new_gene = NewGeneWithTrait(g.Traits[inn.NewTraitNum], inn.NewWeight,
						sensor, output, false, inn.InnovationNum, 0)

					innovation_found = true
					break
				}
			}

			// The innovation is totally novel
			if !innovation_found {
				// Choose a random trait
				trait_num := rand.Intn(len(g.Traits))
				// Choose the new weight
				new_weight := float64(utils.RandSign()) * rand.Float64() * 10.0
				// read next innovation id
				next_innov_id := pop.getNextInnovationNumberAndIncrement()


				// Create the new gene
				new_gene = NewGeneWithTrait(g.Traits[trait_num], new_weight, sensor, output,
					false, next_innov_id, new_weight)

				// Add the innovation for created link
				new_innov := NewInnovationForLink(sensor.Id, output.Id, next_innov_id,
					new_weight, trait_num)
				pop.addInnovationSynced(new_innov)
			} else if g.hasGene(new_gene) {
				// The gene for already occurred innovation already in this genome.
				// This may happen as result of parent genome mutation in current epoch which is
				// repeated in the child after parent's genome transferred to child during mating
				neat.InfoLog(
					fmt.Sprintf("GENOME: Connect sensors innovation found [%t] in the same genome [%d] for gene: %s\n%s",
						innovation_found, g.Id, new_gene, g))
				return false, nil
			}

			// Now add the new Gene to the Genome
			g.Genes = geneInsert(g.Genes, new_gene)
			link_added = true
		}
	}
	return link_added, nil
}

// Mutate the genome by adding a new link between two random NNodes,
// if NNodes are already connected, keep trying conf.NewLinkTries times
func (g *Genome) mutateAddLink(pop *Population, context *neat.NeatContext) (bool, error) {
	// If the phenotype does not exist, exit on false, print error
	// Note: This should never happen - if it does there is a bug
	if g.Phenotype == nil {
		return false, errors.New("Attempt to add link to genome with no phenotype")
	} else if len(g.Nodes) == 0 {
		return false, errors.New("Genome has no nodes to be connected by new link")
	}

	nodes_len := len(g.Nodes)

	// Decide whether to make link recurrent
	do_recur := false
	if rand.Float64() < context.RecurOnlyProb {
		do_recur = true
	}

	// Find the first non-sensor so that the to-node won't look at sensors as possible destinations
	first_non_sensor := 0
	for _, n := range g.Nodes {
		if n.IsSensor() {
			first_non_sensor++
		} else {
			break
		}
	}

	// Made attempts to find an unconnected pair
	try_count := 0

	// Iterate over nodes and try to add new link
	var node_1, node_2 *network.NNode
	found := false
	for try_count < context.NewLinkTries {
		node_num_1, node_num_2 := 0, 0
		if do_recur {
			// 50% of prob to decide create a recurrent link (node X to node X)
			// 50% of a normal link (node X to node Y)
			loop_recur := false
			if rand.Float64() > 0.5 {
				loop_recur = true
			}
			if loop_recur {
				node_num_1 = first_non_sensor + rand.Intn(nodes_len - first_non_sensor) // only NON SENSOR
				node_num_2 = node_num_1
			} else {
				for node_num_1 == node_num_2 {
					node_num_1 = rand.Intn(nodes_len)
					node_num_2 = first_non_sensor + rand.Intn(nodes_len - first_non_sensor) // only NON SENSOR
				}
			}
		} else {
			for node_num_1 == node_num_2 {
				node_num_1 = rand.Intn(nodes_len)
				node_num_2 = first_non_sensor + rand.Intn(nodes_len - first_non_sensor) // only NON SENSOR
			}
		}

		// get corresponding nodes
		node_1 = g.Nodes[node_num_1]
		node_2 = g.Nodes[node_num_2]

		// See if a link already exists  ALSO STOP AT END OF GENES!!!!
		link_exists := false
		if node_2.IsSensor() {
			// Don't allow SENSORS to get input
			link_exists = true
		} else {
			for _, gene := range g.Genes {
				if gene.Link.InNode.Id == node_1.Id &&
					gene.Link.OutNode.Id == node_2.Id &&
					gene.Link.IsRecurrent == do_recur {
					// link already exists
					link_exists = true;
					break;
				}

			}
		}

		if !link_exists {
			// These are used to avoid getting stuck in an infinite loop checking for recursion
			// Note that we check for recursion to control the frequency of adding recurrent links rather
			// than to prevent any particular kind of error
			thresh := nodes_len * nodes_len
			count := 0
			recur_flag := g.Phenotype.IsRecurrent(node_1.PhenotypeAnalogue, node_2.PhenotypeAnalogue, &count, thresh)

			// NOTE: A loop doesn't really matter - just debug output it
			if count > thresh {
				neat.DebugLog(
					fmt.Sprintf("GENOME: LOOP DETECTED DURING A RECURRENCY CHECK -> " +
						"node in: %s <-> node out: %s", node_1.PhenotypeAnalogue, node_2.PhenotypeAnalogue))
			}

			// Make sure it finds the right kind of link (recurrent or not)
			if (!recur_flag && do_recur) || (recur_flag && !do_recur) {
				try_count++
			} else {
				// The open link found
				try_count = context.NewLinkTries
				found = true
			}
		} else {
			try_count++
		}

	}
	// Continue only if an open link was found
	if found {
		var new_gene *Gene
		// Check to see if this innovation already occurred in the population
		innovation_found := false
		for _, inn := range pop.Innovations {
			// match the innovation in the innovations list
			if inn.innovationType == newLinkInnType &&
				inn.InNodeId == node_1.Id &&
				inn.OutNodeId == node_2.Id &&
				inn.IsRecurrent == do_recur {

				// Create new gene
				new_gene = NewGeneWithTrait(g.Traits[inn.NewTraitNum], inn.NewWeight, node_1, node_2, do_recur, inn.InnovationNum, 0)

				innovation_found = true
				break
			}
		}
		// The innovation is totally novel
		if !innovation_found {
			// Choose a random trait
			trait_num := rand.Intn(len(g.Traits))
			// Choose the new weight
			new_weight := float64(utils.RandSign()) * rand.Float64() * 10.0
			// read next innovation id
			next_innov_id := pop.getNextInnovationNumberAndIncrement()

			// Create the new gene
			new_gene = NewGeneWithTrait(g.Traits[trait_num], new_weight, node_1, node_2,
				do_recur, next_innov_id, new_weight)

			// Add the innovation
			new_innov := NewInnovationForRecurrentLink(node_1.Id, node_2.Id, next_innov_id,
				new_weight, trait_num, do_recur)
			pop.addInnovationSynced(new_innov)
		} else if g.hasGene(new_gene) {
			// The gene for already occurred innovation already in this genome.
			// This may happen as result of parent genome mutation in current epoch which is
			// repeated in the child after parent's genome transferred to child during mating
			neat.InfoLog(
				fmt.Sprintf("GENOME: Mutate add link innovation found [%t] in the same genome [%d] for gene: %s\n%s",
					innovation_found, g.Id, new_gene, g))
			return false, nil
		}

		// sanity check
		if new_gene.Link.InNode.Id == new_gene.Link.OutNode.Id && !do_recur {
			neat.DebugLog(fmt.Sprintf("Recurent link created when recurency is not enabled: %s", new_gene))
			return false, errors.New(fmt.Sprintf("GENOME: Wrong gene created!\n%s", g))
		}

		// Now add the new Gene to the Genome
		g.Genes = geneInsert(g.Genes, new_gene)
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
		try_count := 0
		// Alternative uniform random choice of genes. When the genome is not tiny, it is safe to choose randomly.
		for try_count < 20 && !found {
			gene_num := rand.Intn(len(g.Genes))
			gene = g.Genes[gene_num]
			if gene.IsEnabled && gene.Link.InNode.NeuronType != network.BiasNeuron {
				found = true
			}
			try_count++
		}
	}
	if !found {
		// Failed to find appropriate gene
		return false, nil
	}

	gene.IsEnabled = false;

	// Extract the link
	link := gene.Link
	// Extract the weight
	old_weight := link.Weight
	// Get the old link's trait
	trait := link.Trait

	// Extract the nodes
	in_node, out_node := link.InNode, link.OutNode
	if in_node == nil || out_node == nil {
		return false, errors.New(
			fmt.Sprintf("Genome:mutateAddNode: Anomalous link found with either IN or OUT node not set. %s", link))
	}

	var new_gene_1, new_gene_2 *Gene
	var new_node *network.NNode

	// Check to see if this innovation already occurred in the population
	innovation_found := false
	for _, inn := range pop.Innovations {
		/* We check to see if an innovation already occurred that was:
		 	-A new node
		 	-Stuck between the same nodes as were chosen for this mutation
		 	-Splitting the same gene as chosen for this mutation
		 If so, we know this mutation is not a novel innovation in this generation
		 so we make it match the original, identical mutation which occurred
		 elsewhere in the population by coincidence */
		if inn.innovationType == newNodeInnType &&
			inn.InNodeId == in_node.Id &&
			inn.OutNodeId == out_node.Id &&
			inn.OldInnovNum == gene.InnovationNum {

			// Create the new NNode
			new_node = network.NewNNode(inn.NewNodeId, network.HiddenNeuron)
			// By convention, it will point to the first trait
			// Note: In future may want to change this
			new_node.Trait = g.Traits[0]

			// Create the new Genes
			new_gene_1 = NewGeneWithTrait(trait, 1.0, in_node, new_node, link.IsRecurrent, inn.InnovationNum, 0)
			new_gene_2 = NewGeneWithTrait(trait, old_weight, new_node, out_node, false, inn.InnovationNum2, 0)

			innovation_found = true
			break
		}
	}
	// The innovation is totally novel
	if !innovation_found {
		// Get the current node id with post increment
		new_node_id := int(pop.getNextNodeIdAndIncrement())

		// Create the new NNode
		new_node = network.NewNNode(new_node_id, network.HiddenNeuron)
		// By convention, it will point to the first trait
		new_node.Trait = g.Traits[0]
		// Set node activation function as random from a list of types registered with context
		if act_type, err := context.RandomNodeActivationType(); err != nil {
			return false, err
		} else {
			new_node.ActivationType = act_type
		}

		// get the next innovation id for gene 1
		gene_innov_1 := pop.getNextInnovationNumberAndIncrement()
		// create gene with the current gene innovation
		new_gene_1 = NewGeneWithTrait(trait, 1.0, in_node, new_node, link.IsRecurrent, gene_innov_1, 0);

		// get the next innovation id for gene 2
		gene_innov_2 := pop.getNextInnovationNumberAndIncrement()
		// create the second gene with this innovation incremented
		new_gene_2 = NewGeneWithTrait(trait, old_weight, new_node, out_node, false, gene_innov_2, 0);

		// Store innovation
		innov := NewInnovationForNode(in_node.Id, out_node.Id, gene_innov_1, gene_innov_2, new_node.Id, gene.InnovationNum)
		pop.addInnovationSynced(innov)
	} else if g.hasNode(new_node) {
		// The same add node innovation occurred in the same genome (parent) - just skip.
		// This may happen when parent of this organism experienced the same mutation in current epoch earlier
		// and after that parent's genome was duplicated to child by mating and the same mutation parameters
		// was selected again (in_node.Id, out_node.Id, gene.InnovationNum). As result the innovation with given
		// parameters will be found and new node will be created with ID which alredy exists in child genome.
		// If proceed than we will have duplicated Node and genes - so we're skipping this.
		neat.InfoLog(
			fmt.Sprintf("GENOME: Add node innovation found [%t] in the same genome [%d] for node [%d]\n%s",
				innovation_found, g.Id, new_node.Id, g))
		return false, nil
	}


	// Now add the new NNode and new Genes to the Genome
	g.Genes = geneInsert(g.Genes, new_gene_1)
	g.Genes = geneInsert(g.Genes, new_gene_2)
	g.Nodes = nodeInsert(g.Nodes, new_node)

	return true, nil
}

// Adds Gaussian noise to link weights either GAUSSIAN or COLD_GAUSSIAN (from zero).
// The COLD_GAUSSIAN means ALL connection weights will be given completely new values
func (g *Genome) mutateLinkWeights(power, rate float64, mutation_type mutatorType) (bool, error) {
	if len(g.Genes) == 0 {
		return false, errors.New("Genome has no genes")
	}

	// Once in a while really shake things up
	severe := false
	if rand.Float64() > 0.5 {
		severe = true
	}

	// Go through all the Genes and perturb their link's weights
	num, gene_total := 0.0, float64(len(g.Genes))
	end_part := gene_total * 0.8
	var gauss_point, cold_gauss_point float64

	for _, gene := range g.Genes {
		// The following if determines the probabilities of doing cold gaussian
		// mutation, meaning the probability of replacing a link weight with
		// another, entirely random weight. It is meant to bias such mutations
		// to the tail of a genome, because that is where less time-tested genes
		// reside. The gauss_point and cold_gauss_point represent values above
		// which a random float will signify that kind of mutation.
		if severe {
			gauss_point = 0.3
			cold_gauss_point = 0.1
		} else if gene_total >= 10.0 && num > end_part {
			gauss_point = 0.5  // Mutate by modification % of connections
			cold_gauss_point = 0.3 // Mutate the rest by replacement % of the time
		} else {
			// Half the time don't do any cold mutations
			if rand.Float64() > 0.5 {
				gauss_point = 1.0 - rate
				cold_gauss_point = gauss_point - 0.1
			} else {
				gauss_point = 1.0 - rate
				cold_gauss_point = gauss_point // no cold mutation possible (see later)
			}
		}

		rand_val := float64(utils.RandSign()) * rand.Float64() * power
		if mutation_type == gaussianMutator {
			rand_choice := rand.Float64()
			if rand_choice > gauss_point {
				gene.Link.Weight += rand_val
			} else if rand_choice > cold_gauss_point {
				gene.Link.Weight = rand_val
			}
		} else if mutation_type == goldGaussianMutator {
			gene.Link.Weight = rand_val
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
		return false, errors.New("Genome has no traits")
	}
	// Choose a random trait number
	trait_num := rand.Intn(len(g.Traits))

	// Retrieve the trait and mutate it
	g.Traits[trait_num].Mutate(context.TraitMutationPower, context.TraitParamMutProb)

	return true, nil
}

// This chooses a random gene, extracts the link from it and re-points the link to a random trait
func (g *Genome) mutateLinkTrait(times int) (bool, error) {
	if len(g.Traits) == 0 || len(g.Genes) == 0 {
		return false, errors.New("Genome has either no traits od genes")
	}
	for loop := 0; loop < times; loop++ {
		// Choose a random trait number
		trait_num := rand.Intn(len(g.Traits))

		// Choose a random link number
		gene_num := rand.Intn(len(g.Genes))

		// set the link to point to the new trait
		g.Genes[gene_num].Link.Trait = g.Traits[trait_num]

	}
	return true, nil
}

// This chooses a random node and re-points the node to a random trait specified number of times
func (g *Genome) mutateNodeTrait(times int) (bool, error) {
	if len(g.Traits) == 0 || len(g.Nodes) == 0 {
		return false, errors.New("Genome has either no traits or nodes")
	}
	for loop := 0; loop < times; loop++ {
		// Choose a random trait number
		trait_num := rand.Intn(len(g.Traits))

		// Choose a random node number
		node_num := rand.Intn(len(g.Nodes))

		// set the node to point to the new trait
		g.Nodes[node_num].Trait = g.Traits[trait_num]
	}
	return true, nil
}

// Toggle genes from enable on to enable off or vice versa.  Do it specified number of times.
func (g *Genome) mutateToggleEnable(times int) (bool, error) {
	if len(g.Genes) == 0 {
		return false, errors.New("Genome has no genes to toggle")
	}
	for loop := 0; loop < times; loop++ {
		// Choose a random gene number
		gene_num := rand.Intn(len(g.Genes))

		gene := g.Genes[gene_num]
		if gene.IsEnabled {
			// We need to make sure that another gene connects out of the in-node.
			// Because if not a section of network will break off and become isolated.
			for _, check_gene := range g.Genes {
				if check_gene.Link.InNode.Id == gene.Link.InNode.Id &&
					check_gene.IsEnabled && check_gene.InnovationNum != gene.InnovationNum {
					gene.IsEnabled = false
					break
				}
			}
		} else {
			gene.IsEnabled = true
		}

	}
	return true, nil
}
// Finds first disabled gene and enable it
func (g *Genome) mutateGeneReenable() (bool, error) {
	if len(g.Genes) == 0 {
		return false, errors.New("Genome has no genes to re-enable")
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
		res, err = g.mutateGeneReenable();
	}
	return res, err
}

/* ****** MATING METHODS ***** */

// This method mates this Genome with another Genome g. For every point in each Genome, where each Genome shares
// the innovation number, the Gene is chosen randomly from either parent.  If one parent has an innovation absent in
// the other, the baby may inherit the innovation if it is from the more fit parent.
// The new Genome is given the id in the genomeid argument.
func (gen *Genome) mateMultipoint(og *Genome, genomeid int, fitness1, fitness2 float64) (*Genome, error) {
	// Check if genomes has equal number of traits
	if len(gen.Traits) != len(og.Traits) {
		return nil, errors.New(fmt.Sprintf("Genomes has different traits count, %d != %d", len(gen.Traits), len(og.Traits)))
	}

	// First, average the Traits from the 2 parents to form the baby's Traits. It is assumed that trait vectors are
	// the same length. In the future, may decide on a different method for trait mating.
	new_traits, err := gen.mateTraits(og)
	if err != nil {
		return nil, err
	}

	// The new genes and nodes created
	new_genes := make([]*Gene, 0)
	new_nodes := make([]*network.NNode, 0)
	child_nodes_map := make(map[int]*network.NNode)

	// NEW: Make sure all sensors and outputs are included (in case some inputs are disconnected)
	for _, curr_node := range og.Nodes {
		if curr_node.NeuronType == network.InputNeuron ||
			curr_node.NeuronType == network.BiasNeuron ||
			curr_node.NeuronType == network.OutputNeuron {
			node_trait_num := 0
			if curr_node.Trait != nil {
				node_trait_num = curr_node.Trait.Id - gen.Traits[0].Id
			}
			// Create a new node off the sensor or output
			new_onode := network.NewNNodeCopy(curr_node, new_traits[node_trait_num])

			// Add the new node
			new_nodes = nodeInsert(new_nodes, new_onode)
			child_nodes_map[new_onode.Id] = new_onode
		}
	}

	// Figure out which genome is better. The worse genome should not be allowed to add extra structural baggage.
	// If they are the same, use the smaller one's disjoint and excess genes only.
	p1better := false // Tells if the first genome (this one) has better fitness or not
	if fitness1 > fitness2 ||
		(fitness1 == fitness2 && len(gen.Genes) < len(og.Genes)) {
		p1better = true
	}

	// Now loop through the Genes of each parent
	i1, i2, size1, size2 := 0, 0, len(gen.Genes), len(og.Genes)
	var chosen_gene *Gene
	for i1 < size1 || i2 < size2 {
		skip, disable := false, false

		// choose best gene
		if i1 >= size1 {
			chosen_gene = og.Genes[i2]
			i2++
			if p1better {
				skip = true // Skip excess from the worse genome
			}
		} else if i2 >= size2 {
			chosen_gene = gen.Genes[i1]
			i1++
			if !p1better {
				skip = true // Skip excess from the worse genome
			}
		} else {
			p1gene := gen.Genes[i1]
			p2gene := og.Genes[i2]

			// Extract current innovation numbers
			p1innov := p1gene.InnovationNum
			p2innov := p2gene.InnovationNum

			if p1innov == p2innov {
				if rand.Float64() < 0.5 {
					chosen_gene = p1gene
				} else {
					chosen_gene = p2gene
				}

				// If one is disabled, the corresponding gene in the offspring will likely be disabled
				if !p1gene.IsEnabled || !p2gene.IsEnabled && rand.Float64() < 0.75 {
					disable = true
				}
				i1++
				i2++
			} else if p1innov < p2innov {
				chosen_gene = p1gene
				i1++
				if !p1better {
					skip = true // Skip excess from the worse genome
				}
			} else {
				chosen_gene = p2gene
				i2++
				if p1better {
					skip = true // Skip excess from the worse genome
				}
			}
		}

		// Uncomment this line to let growth go faster (from both parents excesses)
		// skip=false

		// Check to see if the chosen gene conflicts with an already chosen gene i.e. do they represent the same link
		for _, new_gene := range new_genes {
			if new_gene.Link.IsEqualGenetically(chosen_gene.Link) {
				skip = true;
				break;
			}
		}

		// Now add the chosen gene to the baby
		if (!skip) {
			// Check for the nodes, add them if not in the baby Genome already
			in_node := chosen_gene.Link.InNode
			out_node := chosen_gene.Link.OutNode

			// Checking for inode's existence
			var new_in_node *network.NNode
			for _, node := range new_nodes {
				if node.Id == in_node.Id {
					new_in_node = node
					break
				}
			}
			if new_in_node == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				in_node_trait_num := 0
				if in_node.Trait != nil {
					in_node_trait_num = in_node.Trait.Id - gen.Traits[0].Id
				}
				new_in_node = network.NewNNodeCopy(in_node, new_traits[in_node_trait_num])
				new_nodes = nodeInsert(new_nodes, new_in_node)
				child_nodes_map[new_in_node.Id] = new_in_node
			}

			// Checking for onode's existence
			var new_out_node *network.NNode
			for _, node := range new_nodes {
				if node.Id == out_node.Id {
					new_out_node = node
					break
				}
			}
			if new_out_node == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				out_node_trait_num := 0
				if out_node.Trait != nil {
					out_node_trait_num = out_node.Trait.Id - gen.Traits[0].Id
				}
				new_out_node = network.NewNNodeCopy(out_node, new_traits[out_node_trait_num])
				new_nodes = nodeInsert(new_nodes, new_out_node)
				child_nodes_map[new_out_node.Id] = new_out_node
			}


			// Add the Gene
			gene_trait_num := 0
			if chosen_gene.Link.Trait != nil {
				// The subtracted number normalizes depending on whether traits start counting at 1 or 0
				gene_trait_num = chosen_gene.Link.Trait.Id - gen.Traits[0].Id
			}
			newgene := NewGeneCopy(chosen_gene, new_traits[gene_trait_num], new_in_node, new_out_node)
			if disable {
				newgene.IsEnabled = false
			}
			new_genes = append(new_genes, newgene)
		} // end SKIP
	} // end FOR
	// check if parent's MIMO control genes should be inherited
	if len(gen.ControlGenes) != 0 || len(og.ControlGenes) != 0 {
		// MIMO control genes found at least in one parent - append it to child if appropriate
		if extra_nodes, modules := gen.mateModules(child_nodes_map, og); modules != nil {
			if len(extra_nodes) > 0 {
				// append extra IO nodes of MIMO genes not found in child
				new_nodes = append(new_nodes, extra_nodes...)
			}

			// Return modular baby genome
			return NewModularGenome(genomeid, new_traits, new_nodes, new_genes, modules), nil
		}
	}
	// Return plain baby Genome
	return NewGenome(genomeid, new_traits, new_nodes, new_genes), nil
}

// This method mates like multipoint but instead of selecting one or the other when the innovation numbers match,
// it averages their weights.
func (gen *Genome) mateMultipointAvg(og *Genome, genomeid int, fitness1, fitness2 float64) (*Genome, error) {
	// Check if genomes has equal number of traits
	if len(gen.Traits) != len(og.Traits) {
		return nil, errors.New(fmt.Sprintf("Genomes has different traits count, %d != %d", len(gen.Traits), len(og.Traits)))
	}

	// First, average the Traits from the 2 parents to form the baby's Traits. It is assumed that trait vectors are
	// the same length. In the future, may decide on a different method for trait mating.
	new_traits, err := gen.mateTraits(og)
	if err != nil {
		return nil, err
	}


	// The new genes and nodes created
	new_genes := make([]*Gene, 0)
	new_nodes := make([]*network.NNode, 0)
	child_nodes_map := make(map[int]*network.NNode)

	// NEW: Make sure all sensors and outputs are included (in case some inputs are disconnected)
	for _, curr_node := range og.Nodes {
		if curr_node.NeuronType == network.InputNeuron ||
			curr_node.NeuronType == network.BiasNeuron ||
			curr_node.NeuronType == network.OutputNeuron {
			node_trait_num := 0
			if curr_node.Trait != nil {
				node_trait_num = curr_node.Trait.Id - gen.Traits[0].Id
			}
			// Create a new node off the sensor or output
			new_onode := network.NewNNodeCopy(curr_node, new_traits[node_trait_num])

			// Add the new node
			new_nodes = nodeInsert(new_nodes, new_onode)
			child_nodes_map[new_onode.Id] = new_onode
		}
	}

	// Figure out which genome is better. The worse genome should not be allowed to add extra structural baggage.
	// If they are the same, use the smaller one's disjoint and excess genes only.
	p1better := false // Tells if the first genome (this one) has better fitness or not
	if fitness1 > fitness2 ||
		(fitness1 == fitness2 && len(gen.Genes) < len(og.Genes)) {
		p1better = true
	}

	// Set up the avgene - this Gene is used to hold the average of the two genes to be averaged
	avg_gene := NewGeneWithTrait(nil, 0.0, nil, nil, false, 0, 0.0);

	// Now loop through the Genes of each parent
	i1, i2, size1, size2 := 0, 0, len(gen.Genes), len(og.Genes)
	var chosen_gene *Gene
	for i1 < size1 || i2 < size2 {
		skip := false
		avg_gene.IsEnabled = true // Default to enabled

		// choose best gene
		if i1 >= size1 {
			chosen_gene = og.Genes[i2]
			i2++
			if p1better {
				skip = true // Skip excess from the worse genome
			}
		} else if i2 >= size2 {
			chosen_gene = gen.Genes[i1]
			i1++
			if !p1better {
				skip = true // Skip excess from the worse genome
			}
		} else {
			p1gene := gen.Genes[i1]
			p2gene := og.Genes[i2]

			// Extract current innovation numbers
			p1innov := p1gene.InnovationNum
			p2innov := p2gene.InnovationNum

			if p1innov == p2innov {
				// Average them into the avg_gene
				if rand.Float64() > 0.5 {
					avg_gene.Link.Trait = p1gene.Link.Trait
				} else {
					avg_gene.Link.Trait = p2gene.Link.Trait
				}
				avg_gene.Link.Weight = (p1gene.Link.Weight + p2gene.Link.Weight) / 2.0 // WEIGHTS AVERAGED HERE

				if rand.Float64() > 0.5 {
					avg_gene.Link.InNode = p1gene.Link.InNode
				} else {
					avg_gene.Link.InNode = p2gene.Link.InNode
				}
				if rand.Float64() > 0.5 {
					avg_gene.Link.OutNode = p1gene.Link.OutNode
				} else {
					avg_gene.Link.OutNode = p2gene.Link.OutNode
				}
				if rand.Float64() > 0.5 {
					avg_gene.Link.IsRecurrent = p1gene.Link.IsRecurrent
				} else {
					avg_gene.Link.IsRecurrent = p2gene.Link.IsRecurrent
				}

				avg_gene.InnovationNum = p1innov
				avg_gene.MutationNum = (p1gene.MutationNum + p2gene.MutationNum) / 2.0
				if !p1gene.IsEnabled || !p2gene.IsEnabled && rand.Float64() < 0.75 {
					avg_gene.IsEnabled = false
				}

				chosen_gene = avg_gene
				i1++
				i2++
			} else if p1innov < p2innov {
				chosen_gene = p1gene
				i1++
				if !p1better {
					skip = true // Skip excess from the worse genome
				}
			} else {
				chosen_gene = p2gene
				i2++
				if p1better {
					skip = true // Skip excess from the worse genome
				}
			}
		}

		// Uncomment this line to let growth go faster (from both parents excesses)
		// skip=false

		// Check to see if the chosen gene conflicts with an already chosen gene i.e. do they represent the same link
		for _, new_gene := range new_genes {
			if new_gene.Link.IsEqualGenetically(chosen_gene.Link) {
				skip = true;
				break;
			}
		}

		if (!skip) {
			// Now add the chosen gene to the baby

			// Check for the nodes, add them if not in the baby Genome already
			in_node := chosen_gene.Link.InNode
			out_node := chosen_gene.Link.OutNode

			// Checking for inode's existence
			var new_in_node *network.NNode
			for _, node := range new_nodes {
				if node.Id == in_node.Id {
					new_in_node = node
					break
				}
			}
			if new_in_node == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				in_node_trait_num := 0
				if in_node.Trait != nil {
					in_node_trait_num = in_node.Trait.Id - gen.Traits[0].Id
				}
				new_in_node = network.NewNNodeCopy(in_node, new_traits[in_node_trait_num])
				new_nodes = nodeInsert(new_nodes, new_in_node)
				child_nodes_map[new_in_node.Id] = new_in_node
			}

			// Checking for onode's existence
			var new_out_node *network.NNode
			for _, node := range new_nodes {
				if node.Id == out_node.Id {
					new_out_node = node
					break
				}
			}
			if new_out_node == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				out_node_trait_num := 0
				if out_node.Trait != nil {
					out_node_trait_num = out_node.Trait.Id - gen.Traits[0].Id
				}
				new_out_node = network.NewNNodeCopy(out_node, new_traits[out_node_trait_num])
				new_nodes = nodeInsert(new_nodes, new_out_node)
				child_nodes_map[new_out_node.Id] = new_out_node
			}

			// Add the Gene
			gene_trait_num := 0
			if chosen_gene.Link.Trait != nil {
				// The subtracted number normalizes depending on whether traits start counting at 1 or 0
				gene_trait_num = chosen_gene.Link.Trait.Id - gen.Traits[0].Id
			}
			new_gene := NewGeneCopy(chosen_gene, new_traits[gene_trait_num], new_in_node, new_out_node)
			new_genes = append(new_genes, new_gene)
		} // end SKIP
	} // end FOR
	// check if parent's MIMO control genes should be inherited
	if len(gen.ControlGenes) != 0 || len(og.ControlGenes) != 0 {
		// MIMO control genes found at least in one parent - append it to child if appropriate
		if extra_nodes, modules := gen.mateModules(child_nodes_map, og); modules != nil {
			if len(extra_nodes) > 0 {
				// append extra IO nodes of MIMO genes not found in child
				new_nodes = append(new_nodes, extra_nodes...)
			}

			// Return modular baby genome
			return NewModularGenome(genomeid, new_traits, new_nodes, new_genes, modules), nil
		}
	}
	// Return plain baby Genome
	return NewGenome(genomeid, new_traits, new_nodes, new_genes), nil
}

// This method is similar to a standard single point CROSSOVER operator. Traits are averaged as in the previous two
// mating methods. A Gene is chosen in the smaller Genome for splitting. When the Gene is reached, it is averaged with
// the matching Gene from the larger Genome, if one exists. Then every other Gene is taken from the larger Genome.
func (gen *Genome) mateSinglepoint(og *Genome, genomeid int) (*Genome, error) {
	// Check if genomes has equal number of traits
	if len(gen.Traits) != len(og.Traits) {
		return nil, errors.New(fmt.Sprintf("Genomes has different traits count, %d != %d", len(gen.Traits), len(og.Traits)))
	}

	// First, average the Traits from the 2 parents to form the baby's Traits. It is assumed that trait vectors are
	// the same length. In the future, may decide on a different method for trait mating.
	new_traits, err := gen.mateTraits(og)
	if err != nil {
		return nil, err
	}

	// The new genes and nodes created
	new_genes := make([]*Gene, 0)
	new_nodes := make([]*network.NNode, 0)
	child_nodes_map := make(map[int]*network.NNode)

	// NEW: Make sure all sensors and outputs are included (in case some inputs are disconnected)
	for _, curr_node := range og.Nodes {
		if curr_node.NeuronType == network.InputNeuron ||
			curr_node.NeuronType == network.BiasNeuron ||
			curr_node.NeuronType == network.OutputNeuron {
			node_trait_num := 0
			if curr_node.Trait != nil {
				node_trait_num = curr_node.Trait.Id - gen.Traits[0].Id
			}
			// Create a new node off the sensor or output
			new_onode := network.NewNNodeCopy(curr_node, new_traits[node_trait_num])

			// Add the new node
			new_nodes = nodeInsert(new_nodes, new_onode)
			child_nodes_map[new_onode.Id] = new_onode
		}
	}

	// Set up the avg_gene - this Gene is used to hold the average of the two genes to be averaged
	avg_gene := NewGeneWithTrait(nil, 0.0, nil, nil, false, 0, 0.0);

	p1stop, p2stop, stopper, crosspoint := 0, 0, 0, 0
	var p1genes, p2genes []*Gene
	size1, size2 := len(gen.Genes), len(og.Genes)
	if size1 < size2 {
		crosspoint = rand.Intn(size1)
		p1stop = size1
		p2stop = size2
		stopper = size2
		p1genes = gen.Genes
		p2genes = og.Genes
	} else {
		crosspoint = rand.Intn(size2)
		p1stop = size2
		p2stop = size1
		stopper = size1
		p1genes = og.Genes
		p2genes = gen.Genes
	}

	var chosen_gene *Gene
	gene_counter, i1, i2 := 0, 0, 0
	// Now move through the Genes of each parent until both genomes end
	for i2 < stopper {
		skip := false
		avg_gene.IsEnabled = true  // Default to true
		if i1 == p1stop {
			chosen_gene = p2genes[i2]
			i2++
		} else if i2 == p2stop {
			chosen_gene = p1genes[i1]
			i1++
		} else {
			p1gene := p1genes[i1]
			p2gene := p2genes[i2]

			// Extract current innovation numbers
			p1innov := p1gene.InnovationNum
			p2innov := p2gene.InnovationNum

			if p1innov == p2innov {
				//Pick the chosen gene depending on whether we've crossed yet
				if gene_counter < crosspoint {
					chosen_gene = p1gene
				} else if gene_counter > crosspoint {
					chosen_gene = p2gene
				} else {
					// We are at the crosspoint here - average genes into the avgene
					if rand.Float64() > 0.5 {
						avg_gene.Link.Trait = p1gene.Link.Trait
					} else {
						avg_gene.Link.Trait = p2gene.Link.Trait
					}
					avg_gene.Link.Weight = (p1gene.Link.Weight + p2gene.Link.Weight) / 2.0 // WEIGHTS AVERAGED HERE

					if rand.Float64() > 0.5 {
						avg_gene.Link.InNode = p1gene.Link.InNode
					} else {
						avg_gene.Link.InNode = p2gene.Link.InNode
					}
					if rand.Float64() > 0.5 {
						avg_gene.Link.OutNode = p1gene.Link.OutNode
					} else {
						avg_gene.Link.OutNode = p2gene.Link.OutNode
					}
					if rand.Float64() > 0.5 {
						avg_gene.Link.IsRecurrent = p1gene.Link.IsRecurrent
					} else {
						avg_gene.Link.IsRecurrent = p2gene.Link.IsRecurrent
					}

					avg_gene.InnovationNum = p1innov
					avg_gene.MutationNum = (p1gene.MutationNum + p2gene.MutationNum) / 2.0
					if !p1gene.IsEnabled || !p2gene.IsEnabled && rand.Float64() < 0.75 {
						avg_gene.IsEnabled = false
					}

					chosen_gene = avg_gene
				}
				i1++
				i2++
				gene_counter++
			} else if p1innov < p2innov {
				if (gene_counter < crosspoint) {
					chosen_gene = p1gene
					i1++
					gene_counter++
				} else {
					chosen_gene = p2gene
					i2++
				}
			} else {
				// p2innov < p1innov
				i2++
				// Special case: we need to skip to the next iteration
				// because this Gene is before the crosspoint on the wrong Genome
				skip = true
			}
		}
		// Check to see if the chosen gene conflicts with an already chosen gene i.e. do they represent the same link
		for _, new_gene := range new_genes {
			if new_gene.Link.IsEqualGenetically(chosen_gene.Link) {
				skip = true;
				break;
			}
		}

		//Now add the chosen gene to the baby
		if (!skip) {
			// Check for the nodes, add them if not in the baby Genome already
			in_node := chosen_gene.Link.InNode
			out_node := chosen_gene.Link.OutNode

			// Checking for inode's existence
			var new_in_node *network.NNode
			for _, node := range new_nodes {
				if node.Id == in_node.Id {
					new_in_node = node
					break
				}
			}
			if new_in_node == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				in_node_trait_num := 0
				if in_node.Trait != nil {
					in_node_trait_num = in_node.Trait.Id - gen.Traits[0].Id
				}
				new_in_node = network.NewNNodeCopy(in_node, new_traits[in_node_trait_num])
				new_nodes = nodeInsert(new_nodes, new_in_node)
				child_nodes_map[new_in_node.Id] = new_in_node
			}

			// Checking for onode's existence
			var new_out_node *network.NNode
			for _, node := range new_nodes {
				if node.Id == out_node.Id {
					new_out_node = node
					break
				}
			}
			if new_out_node == nil {
				// Here we know the node doesn't exist so we have to add it normalized trait
				// number for new NNode
				out_node_trait_num := 0
				if out_node.Trait != nil {
					out_node_trait_num = out_node.Trait.Id - gen.Traits[0].Id
				}
				new_out_node = network.NewNNodeCopy(out_node, new_traits[out_node_trait_num])
				new_nodes = nodeInsert(new_nodes, new_out_node)
				child_nodes_map[new_out_node.Id] = new_out_node
			}

			// Add the Gene
			gene_trait_num := 0
			if chosen_gene.Link.Trait != nil {
				// The subtracted number normalizes depending on whether traits start counting at 1 or 0
				gene_trait_num = chosen_gene.Link.Trait.Id - gen.Traits[0].Id
			}
			new_gene := NewGeneCopy(chosen_gene, new_traits[gene_trait_num], new_in_node, new_out_node)
			new_genes = append(new_genes, new_gene)
		}// end SKIP
	} // end FOR
	// check if parent's MIMO control genes should be inherited
	if len(gen.ControlGenes) != 0 || len(og.ControlGenes) != 0 {
		// MIMO control genes found at least in one parent - append it to child if appropriate
		if extra_nodes, modules := gen.mateModules(child_nodes_map, og); modules != nil {
			if len(extra_nodes) > 0 {
				// append extra IO nodes of MIMO genes not found in child
				new_nodes = append(new_nodes, extra_nodes...)
			}

			// Return modular baby genome
			return NewModularGenome(genomeid, new_traits, new_nodes, new_genes, modules), nil
		}
	}
	// Return plain baby Genome
	return NewGenome(genomeid, new_traits, new_nodes, new_genes), nil
}

// Builds an array of modules to be added to the child during crossover.
// If any or both parents has module and at least one modular endpoint node already inherited by child genome than make
// sure that child get all associated module nodes
func (g *Genome) mateModules(child_nodes map[int]*network.NNode, og *Genome) ([]*network.NNode, []*MIMOControlGene) {
	parent_modules := make([]*MIMOControlGene, 0)
	g_modules := findModulesIntersection(child_nodes, g.ControlGenes)
	if len(g_modules) > 0 {
		parent_modules = append(parent_modules, g_modules...)
	}
	og_modules := findModulesIntersection(child_nodes, og.ControlGenes)
	if len(og_modules) > 0 {
		parent_modules = append(parent_modules, og_modules...)
	}
	if len(parent_modules) == 0 {
		return nil, nil
	}

	// collect IO nodes from all included modules and add return it as extra ones
	extra_nodes := make([]*network.NNode, 0)
	for _, cg := range parent_modules {
		for _, n := range cg.ioNodes {
			if _, ok := child_nodes[n.Id]; !ok {
				// not found in known child nodes - collect it
				extra_nodes = append(extra_nodes, n)
			}
		}
	}

	return extra_nodes, parent_modules
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
	new_traits := make([]*neat.Trait, len(g.Traits))
	var err error
	for i, tr := range g.Traits {
		new_traits[i], err = neat.NewTraitAvrg(tr, og.Traits[i]) // construct by averaging
		if err != nil {
			return nil, err
		}
	}
	return new_traits, nil
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
	num_disjoint, num_excess, mut_diff_total, num_matching := 0.0, 0.0, 0.0, 0.0
	size1, size2 := len(g.Genes), len(og.Genes)
	max_genome_size := size2
	if size1 > size2 {
		max_genome_size = size1
	}
	var gene1, gene2 *Gene
	for i, i1, i2 := 0, 0, 0; i < max_genome_size; i++ {
		if i1 >= size1 {
			num_excess += 1.0
			i2++
		} else if i2 >= size2 {
			num_excess += 1.0
			i1++
		} else {
			gene1 = g.Genes[i1]
			gene2 = og.Genes[i2]
			p1innov := gene1.InnovationNum
			p2innov := gene2.InnovationNum

			if p1innov == p2innov {
				num_matching += 1.0
				mut_diff := math.Abs(gene1.MutationNum - gene2.MutationNum)
				mut_diff_total += mut_diff
				i1++
				i2++
			} else if p1innov < p2innov {
				i1++
				num_disjoint += 1.0
			} else if p2innov < p1innov {
				i2++
				num_disjoint += 1.0
			}
		}
	}

	//fmt.Printf("num_disjoint: %.f num_excess: %.f mut_diff_total: %.f num_matching: %.f\n", num_disjoint, num_excess, mut_diff_total, num_matching)

	// Return the compatibility number using compatibility formula
	// Note that mut_diff_total/num_matching gives the AVERAGE difference between mutation_nums for any two matching
	// Genes in the Genome. Look at disjointedness and excess in the absolute (ignoring size)
	comp := context.DisjointCoeff * num_disjoint + context.ExcessCoeff * num_excess +
		context.MutdiffCoeff * (mut_diff_total / num_matching)

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
	list1_count, list2_count := len(g.Genes), len(og.Genes)
	// First test edge cases
	if list1_count == 0 && list2_count == 0 {
		// Both lists are empty! No disparities, therefore the genomes are compatible!
		return 0.0
	}
	if list1_count == 0 {
		// All list2 genes are excess.
		return float64(list2_count) * context.ExcessCoeff
	}

	if list2_count == 0 {
		// All list1 genes are excess.
		return float64(list1_count) * context.ExcessCoeff
	}

	excess_genes_switch, num_matching := 0, 0
	compatibility, mut_diff := 0.0, 0.0
	list1_idx, list2_idx := list1_count - 1, list2_count - 1
	gene1, gene2 := g.Genes[list1_idx], og.Genes[list2_idx]

	for {
		if gene2.InnovationNum > gene1.InnovationNum {
			// Most common test case(s) at top for efficiency.
			if excess_genes_switch == 3 {
				// No more excess genes. Therefore this mismatch is disjoint.
				compatibility += context.DisjointCoeff
			} else if excess_genes_switch == 2 {
				// Another excess gene on genome 2.
				compatibility += context.ExcessCoeff
			} else if excess_genes_switch == 1 {
				// We have found the first non-excess gene.
				excess_genes_switch = 3
				compatibility += context.DisjointCoeff
			} else {
				// First gene is excess, and is on genome 2.
				excess_genes_switch = 2
				compatibility += context.ExcessCoeff
			}

			// Move to the next gene in list2.
			list2_idx--
		} else if gene1.InnovationNum == gene2.InnovationNum {
			// No more excess genes. It's quicker to set this every time than to test if is not yet 3.
			excess_genes_switch = 3

			// Matching genes. Increase compatibility by MutationNum difference * coeff.
			mut_diff += math.Abs(gene1.MutationNum - gene2.MutationNum)
			num_matching++

			// Move to the next gene in both lists.
			list1_idx--
			list2_idx--
		} else {
			// Most common test case(s) at top for efficiency.
			if excess_genes_switch == 3 {
				// No more excess genes. Therefore this mismatch is disjoint.
				compatibility += context.DisjointCoeff
			} else if (excess_genes_switch == 1) {
				// Another excess gene on genome 1.
				compatibility += context.ExcessCoeff
			} else if excess_genes_switch == 2 {
				// We have found the first non-excess gene.
				excess_genes_switch = 3
				compatibility += context.DisjointCoeff
			} else {
				// First gene is excess, and is on genome 1.
				excess_genes_switch = 1
				compatibility += context.ExcessCoeff
			}

			// Move to the next gene in list1.
			list1_idx--
		}

		// Check if we have reached the end of one (or both) of the lists. If we have reached the end of both then
		// we execute the first 'if' block - but it doesn't matter since the loop is not entered if both lists have
		// been exhausted.
		if list1_idx < 0 {
			// All remaining list2 genes are disjoint.
			compatibility += float64(list2_idx + 1) * context.DisjointCoeff
			break

		}

		if list2_idx < 0 {
			// All remaining list1 genes are disjoint.
			compatibility += float64(list1_idx + 1) * context.DisjointCoeff
			break
		}

		gene1, gene2 = g.Genes[list1_idx], og.Genes[list2_idx]
	}
	if num_matching > 0 {
		compatibility += mut_diff * context.MutdiffCoeff / float64(num_matching)
	}
	return compatibility
}

