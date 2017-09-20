package genetics

import (
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat"
	"math/rand"
	"io"
	"fmt"
	"errors"
	"math"
)

// A Genome is the primary source of genotype information used to create  a phenotype.
// It contains 3 major constituents:
// 	1) A Vector of Traits
// 	2) A List of NNodes pointing to a Trait from (1)
// 	3) A List of Genes with Links that point to Traits from (1)
//
// (1) Reserved parameter space for future use.
// (2) NNode specifications.
// (3) Is the primary source of innovation in the evolutionary Genome.
//
// Each Gene in (3) has a marker telling when it arose historically. Thus, these Genes can be used to speciate the
// population, and the list of Genes provide an evolutionary history of innovation and link-building.
type Genome struct {
	// The genome ID
	Id        int
	// The parameters conglomerations
	Traits    []*neat.Trait
	// List of NNodes for the Network
	Nodes     []*network.NNode
	// List of innovation-tracking genes
	Genes     []*Gene

	// Allows Genome to be matched with its Network
	Phenotype *network.Network
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

// Constructor which takes in links (not genes) and creates a Genome
func NewGenomeFromLinks(id int, t []*neat.Trait, n []*network.NNode, links []*network.Link) *Genome {
	gnome := Genome{
		Id:id,
		Traits:t,
		Nodes:n,
		Genes:make([]*Gene, len(links)),
	}
	// Iterate over links and turn them into genes
	for i, l := range links {
		gnome.Genes[i] = NewGeneWithTrait(l.Trait, l.Weight, l.InNode, l.OutNode, l.IsRecurrent, 1.0, 0.0)
	}
	return &gnome
}

// This special constructor creates a Genome with in inputs, out outputs, n out of nmax hidden units, and random
// connectivity.  If rec is true then recurrent connections will be included. The last input is a bias
// link_prob is the probability of a link  */
func NewGenomeRand(new_id, in, out, n, nmax int, recurrent bool, link_prob float64) *Genome {
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
			new_node = network.NewNNodeInPlace(network.SENSOR, ncount, network.INPUT)
		} else {
			new_node = network.NewNNodeInPlace(network.SENSOR, ncount, network.BIAS)
		}
		new_node.Trait = new_trait
		gnome.Nodes = append(gnome.Nodes, new_node)
	}

	// Build the hidden nodes
	for ncount := in + 1; ncount <= in + n; ncount++ {
		new_node = network.NewNNodeInPlace(network.NEURON, ncount, network.HIDDEN)
		new_node.Trait = new_trait
		gnome.Nodes = append(gnome.Nodes, new_node)
	}

	// Build the output nodes
	for ncount := first_output; ncount <= total_nodes; ncount++ {
		new_node = network.NewNNodeInPlace(network.NEURON, ncount, network.OUTPUT)
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
					new_weight := float64(neat.RandPosNeg()) * rand.Float64()
					new_gene = NewGeneWithTrait(new_trait, new_weight, in_node, out_node, flag_recurrent, count, new_weight)

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
func ReadGenome(r io.Reader, id int) (*Genome, error) {
	gnome := Genome{
		Id:id,
		Traits:make([]*neat.Trait, 0),
		Nodes:make([]*network.NNode, 0),
		Genes:make([]*Gene, 0),
	}

	done := false
	var cur_word string
	// Loop until file is finished, parsing each line
	for !done {
		fmt.Fscanf(r, "%s", &cur_word)

		//fmt.Println(cur_word)

		switch cur_word {
		case "genomestart":
			// Read Genome ID and check
			var g_id int
			fmt.Fscanf(r, "%d ", &g_id)
			if g_id != id {
				return nil, errors.New(
					fmt.Sprintf("Id mismatch in genome. Found: %d, expected: %d", g_id, id))
			}

		case "trait":
			// Read a Trait
			new_trait := neat.ReadTrait(r)
			gnome.Traits = append(gnome.Traits, new_trait)

		case "node":
			// Read a NNode
			new_node := network.ReadNNode(r, gnome.Traits)
			gnome.Nodes = append(gnome.Nodes, new_node)

		case "gene":
			// Read a Gene
			new_gene := ReadGene(r, gnome.Traits, gnome.Nodes)
			gnome.Genes = append(gnome.Genes, new_gene)

		case "genomeend":
			// Finish
			done = true

		default:
			// Print to the screen
			fmt.Printf("%s \n", cur_word)
		}
	}
	return &gnome, nil
}

// Writes this genome into provided writer
func (g *Genome) WriteGenome(w io.Writer) {
	fmt.Fprintf(w, "genomestart %d\n", g.Id)

	for _, tr := range g.Traits {
		fmt.Fprint(w, "trait ")
		tr.WriteTrait(w)
		fmt.Fprintln(w, "")
	}

	for _, nd := range g.Nodes {
		fmt.Fprint(w, "node ")
		nd.WriteNode(w)
		fmt.Fprintln(w, "")
	}

	for _, gn := range g.Genes {
		fmt.Fprint(w, "gene ")
		gn.WriteGene(w)
		fmt.Fprintln(w, "")
	}
	fmt.Fprintf(w, "genomeend %d\n", g.Id)
}

// Stringer
func (g *Genome) String() string {
	str := "GENOME START\nNodes:\n"
	for _, n := range g.Nodes {
		n_type := ""
		switch n.NType {
		case network.INPUT:
			n_type = "I"
		case network.OUTPUT:
			n_type = "O"
		case network.BIAS:
			n_type = "B"
		case network.HIDDEN:
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

// Generate a Network phenotype from this Genome with specified id
func (g *Genome) genesis(net_id int) *network.Network {
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
		if n.GenNodeLabel == network.INPUT || n.GenNodeLabel == network.BIAS {
			in_list = append(in_list, new_node)
		} else if n.GenNodeLabel == network.OUTPUT {
			out_list = append(out_list, new_node)
		}

		// Keep track of all nodes in one place for convenience
		all_list = append(all_list, new_node)

		// Have the node specifier point to the node it generated
		n.Analogue = new_node
	}

	if len(g.Genes) == 0 {
		fmt.Println("ALERT : the network built whitout GENES; the result can be unpredictable")
	}

	if len(out_list) == 0 {
		fmt.Println("ALERT : the network whitout OUTPUTS; the result can be unpredictable");
		fmt.Println(g)
	}

	var in_node, out_node *network.NNode
	var cur_link, new_link *network.Link
	// Create the links by iterating through the genes
	for _, gn := range g.Genes {
		// Only create the link if the gene is enabled
		if gn.IsEnabled {
			cur_link = gn.Link
			in_node = cur_link.InNode.Analogue
			out_node = cur_link.OutNode.Analogue

			// NOTE: This line could be run through a recurrency check if desired
			// (no need to in the current implementation of NEAT)
			new_link = network.NewLinkWithTrait(cur_link.Trait, cur_link.Weight, in_node, out_node, cur_link.IsRecurrent)

			// Add link to the connected nodes
			out_node.Incoming = append(out_node.Incoming, new_link)
			in_node.Outgoing = append(in_node.Outgoing, new_link)
		}
	}

	// Create the new network
	new_net := network.NewNetwork(in_list, out_list, all_list, net_id)
	// Attach genotype and phenotype together:
	// new_net point to owner genotype (this)
	// TODO new_net.Genome = g

	// genotype points to owner phenotype (new_net)
	g.Phenotype = new_net

	return new_net
}

// Duplicate this Genome to create a new one with the specified id
func (g *Genome) duplicate(new_id int) *Genome {

	// Duplicate the traits
	traits_dup := make([]*neat.Trait, 0)
	for _, tr := range g.Traits {
		new_trait := neat.NewTraitCopy(tr)
		traits_dup = append(traits_dup, new_trait)
	}

	// Duplicate NNodes
	var assoc_trait *neat.Trait
	nodes_dup := make([]*network.NNode, 0)
	for _, nd := range g.Nodes {
		// First, find the trait that this node points to
		assoc_trait = nil
		if nd.Trait != nil {
			for _, tr := range traits_dup {
				if nd.Trait.Id == tr.Id {
					assoc_trait = tr
					break
				}
			}
		}

		new_node := network.NewNNodeCopy(nd, assoc_trait)
		// Remember this node's old copy
		nd.Duplicate = new_node

		nodes_dup = append(nodes_dup, new_node)
	}

	// Duplicate Genes
	genes_dup := make([]*Gene, 0)
	for _, gn := range g.Genes {
		// First find the nodes connected by the gene's link
		in_node := gn.Link.InNode.Duplicate
		out_node := gn.Link.OutNode.Duplicate

		// Find the trait associated with this gene
		assoc_trait = nil
		if gn.Link.Trait != nil {
			for _, tr := range traits_dup {
				if gn.Link.Trait.Id == tr.Id {
					assoc_trait = tr
					break
				}
			}
		}
		new_gene := NewGeneGeneCopy(gn, assoc_trait, in_node, out_node)
		genes_dup = append(genes_dup, new_gene)
	}

	return NewGenome(new_id, traits_dup, nodes_dup, genes_dup)
}

// For debugging: A number of tests can be run on a genome to check its integrity.
// Note: Some of these tests do not indicate a bug, but rather are meant to be used to detect specific system states.
func (g *Genome) verify() (bool, error) {
	if len(g.Genes) == 0 {
		return false, errors.New("Ganome has no Genes")
	}
	if len(g.Nodes) == 0 {
		return false, errors.New("Ganome has no Nodes")
	}
	if len(g.Traits) == 0 {
		return false, errors.New("Ganome has no Traits")
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
			if gn != gn2 &&
				gn.Link.IsRecurrent == gn2.Link.IsRecurrent &&
				gn.Link.InNode.Id == gn2.Link.InNode.Id &&
				gn.Link.OutNode.Id == gn2.Link.OutNode.Id {
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

/* ******* MUTATORS ******* */

// Mutate the genome by adding a new link between 2 random NNodes
func (g *Genome) mutateAddLink(pop *Population, tries int) bool {
	// TODO Implement this
	return false
}
// Mutate genome by adding a node representation
func (g *Genome) mutateAddNode(pop *Population) bool {
	// TODO Implement this
	return false
}

// Adds Gaussian noise to link weights either GAUSSIAN or COLDGAUSSIAN (from zero)
func (g *Genome) mutateLinkWeights(power, rate float64, mut_type int) {
	// TODO Implement this
}
// Perturb params in one trait
func (g *Genome) mutateRandomTrait() {
	// TODO Implement this
}
// Change random link's trait. Repeat times times
func (g *Genome) mutateLinkTrait(times int) {
	// TODO Implement this
}
// Change random node's trait times
func (g *Genome) mutateNodeTrait(times int) {
	// TODO Implement this
}
// Toggle genes on or off
func (g *Genome) mutateToggleEnable(times int) {
	// TODO Implement th
}
// Find first disabled gene and enable it
func (g *Genome) mutateGeneReenable() {
	// TODO Implement th
}

// Applies all non-structural mutations to this genome
func (g *Genome) mutateAllNonstructural(conf *neat.Neat) {
	if rand.Float64() < conf.MutateRandomTraitProb {
		// mutate random trait
		g.mutateRandomTrait()
	}

	if rand.Float64() < conf.MutateLinkTraitProb {
		// mutate link trait
		g.mutateLinkTrait(1)
	}

	if rand.Float64() < conf.MutateNodeTraitProb {
		// mutate node trait
		g.mutateNodeTrait(1)
	}

	if rand.Float64() < conf.MutateLinkWeightsProb {
		// mutate link weight
		g.mutateLinkWeights(conf.WeightMutPower, 1.0, GAUSSIAN)
	}

	if rand.Float64() < conf.MutateToggleEnableProb {
		// mutate toggle enable
		g.mutateToggleEnable(1)
	}

	if rand.Float64() < conf.MutateGeneReenableProb {
		// mutate gene reenable
		g.mutateGeneReenable();
	}
}

/* ****** MATING METHODS ***** */

// This method mates this Genome with another Genome g. For every point in each Genome, where each Genome shares
// the innovation number, the Gene is chosen randomly from either parent.  If one parent has an innovation absent in
// the other, the baby will inherit the innovation
func (g *Genome) mateMultipoint(og *Genome, genomeid int, fitness1, fitness2 float64) *Genome {
	// TODO implement this
	return nil
}

// This method mates like multipoint but instead of selecting one or the other when the innovation numbers match,
// it averages their weights.
func (g *Genome) mateMultipointAvg(og *Genome, genomeid int, fitness1, fitness2 float64) *Genome {
	// TODO implement this
	return nil
}

// This method is similar to a standard single point CROSSOVER operator. Traits are averaged as in the previous two
// mating methods. A point is chosen in the smaller Genome for crossing with the bigger one.
func (g *Genome) mateSinglepoint(og *Genome, genomeid int) *Genome {
	// TODO implement this
	return nil
}

/* ******** COMPATIBILITY CHECKING METHODS * ********/

// This function gives a measure of compatibility between two Genomes by computing a linear combination of 3
// characterizing variables of their compatibilty. The 3 variables represent PERCENT DISJOINT GENES,
// PERCENT EXCESS GENES, MUTATIONAL DIFFERENCE WITHIN MATCHING GENES. So the formula for compatibility
// is:  disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg
// The 3 coefficients are global system parameters */
func (g *Genome) compatibility(og *Genome, conf *neat.Neat) float64 {
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
				mut_diff :=  math.Abs(gene1.MutationNum - gene2.MutationNum)
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
	comp := conf.DisjointCoeff * num_disjoint + conf.ExcessCoeff * num_excess +
		conf.MutdiffCoeff * (mut_diff_total / num_matching)

	return comp
}




