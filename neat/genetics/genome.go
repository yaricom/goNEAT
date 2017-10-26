package genetics

import (
	"github.com/yaricom/goNEAT/neat/network"
	"github.com/yaricom/goNEAT/neat"
	"math/rand"
	"io"
	"fmt"
	"errors"
	"math"
	"bufio"
	"strings"
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
	gnome := Genome{
		Id:id,
		Traits:make([]*neat.Trait, 0),
		Nodes:make([]*network.NNode, 0),
		Genes:make([]*Gene, 0),
	}

	var g_id int
	// Loop until file is finished, parsing each line
	scanner := bufio.NewScanner(ir)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, " ", 2)
		if len(parts) < 2 {
			return nil, errors.New(fmt.Sprintf("Line: [%s] can not be split when reading Genome", line))
		}
		lr := strings.NewReader(parts[1])

		switch parts[0] {
		case "trait":
			// Read a Trait
			new_trait := neat.ReadTrait(lr)
			gnome.Traits = append(gnome.Traits, new_trait)

		case "node":
			// Read a NNode
			new_node := network.ReadNNode(lr, gnome.Traits)
			gnome.Nodes = append(gnome.Nodes, new_node)

		case "gene":
			// Read a Gene
			new_gene := ReadGene(lr, gnome.Traits, gnome.Nodes)
			gnome.Genes = append(gnome.Genes, new_gene)

		case "genomeend":
			// Read Genome ID
			fmt.Fscanf(lr, "%d", &g_id)
			// check that we have correct genome ID
			if g_id != id {
				return nil, errors.New(
					fmt.Sprintf("Id mismatch in genome. Found: %d, expected: %d", g_id, id))
			}

		case "/*":
			// read all comments and print it
			neat.InfoLog(line)
		}
	}
	return &gnome, nil
}

// Writes this genome into provided writer
func (g *Genome) Write(w io.Writer) {
	fmt.Fprintf(w, "genomestart %d\n", g.Id)

	for _, tr := range g.Traits {
		fmt.Fprint(w, "trait ")
		tr.WriteTrait(w)
		fmt.Fprintln(w, "")
	}

	for _, nd := range g.Nodes {
		fmt.Fprint(w, "node ")
		nd.Write(w)
		fmt.Fprintln(w, "")
	}

	for _, gn := range g.Genes {
		fmt.Fprint(w, "gene ")
		gn.Write(w)
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

// Return id of final NNode in Genome + 1
func (g *Genome) getLastNodeId() (int, error) {
	if len(g.Nodes) > 0 {
		return g.Nodes[len(g.Nodes) - 1].Id + 1, nil
	} else {
		return -1, errors.New("Genome has no nodes")
	}
}

// Return innovation number of last gene in Genome + 1
func (g *Genome) getLastGeneInnovNum() (int64, error) {
	if len(g.Genes) > 0 {
		return g.Genes[len(g.Genes) - 1].InnovationNum + int64(1), nil
	} else {
		return -1, errors.New("Genome has no Genes")
	}
}

// Returns true if this Genome already includes provided node
func (g *Genome) hasNode(node *network.NNode) bool {
	if id, _ := g.getLastNodeId(); node.Id >= id {
		return false
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
	if inn, _ := g.getLastGeneInnovNum(); gene.InnovationNum >= inn {
		return false
	}

	for _, g := range g.Genes {
		if g.Link.InNode.Id == gene.Link.InNode.Id &&
			g.Link.OutNode.Id == gene.Link.OutNode.Id &&
			g.Link.IsRecurrent == gene.Link.IsRecurrent {
			return true
		}
	}
	return false
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
		neat.WarnLog("ALERT : the network built whitout GENES; the result can be unpredictable")
	}

	if len(out_list) == 0 {
		neat.WarnLog("ALERT : the network whitout OUTPUTS; the result can be unpredictable");
		neat.WarnLog(g)
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
		new_gene := NewGeneCopy(gn, assoc_trait, in_node, out_node)
		genes_dup = append(genes_dup, new_gene)
	}

	return NewGenome(new_id, traits_dup, nodes_dup, genes_dup)
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

// Inserts a NNode into a given ordered list of NNodes in ascending order by NNode ID
func nodeInsert(nodes[]*network.NNode, n *network.NNode) []*network.NNode {
	index := len(nodes) // to make sure that greater IDs appended at the end
	for i, node := range nodes {
		if node.Id >= n.Id {
			index = i
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
	for i, gene := range genes {
		if gene.InnovationNum >= g.InnovationNum {
			index = i
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

// Mutate the genome by adding connections to disconnected inputs (sensors).
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
		if n.NType == network.SENSOR {
			sensors = append(sensors, n)
		} else if n.GenNodeLabel == network.OUTPUT {
			outputs = append(outputs, n)
		}
	}

	// eliminate from contention any sensors that are already connected
	disconnected_sensors := make([]*network.NNode, 0)
	for _, sensor := range sensors {
		outputConnections := 0

		// iterate over all genes and count number of output connections from given sensor
		for _, gene := range g.Genes {
			if gene.Link.InNode == sensor && gene.Link.OutNode.GenNodeLabel == network.OUTPUT {
				outputConnections++
			}
		}

		if outputConnections != len(outputs) {
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
				if inn.InnovationType == NEWLINK &&
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
				new_weight := float64(neat.RandPosNeg()) * rand.Float64() * 10.0
				// read curr innovation with post increment
				curr_innov := pop.getInnovationNumberAndIncrement()


				// Create the new gene
				new_gene = NewGeneWithTrait(g.Traits[trait_num], new_weight, sensor, output,
					false, curr_innov, new_weight)

				// Add the innovation for created link
				new_innov := NewInnovationForLink(sensor.Id, output.Id, curr_innov,
					new_weight, trait_num)
				pop.Innovations = append(pop.Innovations, new_innov)
			} else if g.hasGene(new_gene) {
				// The gene for already occurred innovation already in this genome.
				// This may happen as result of parent genome mutation in current epoch which is
				// repeated in the child after parent's genome transferred to child during mating
				neat.DebugLog(
					fmt.Sprintf("GENOME: ALERT: Connect sensors innovation found [%t] in the same genome [%d] for gene: %s\n%s",
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

	// These are used to avoid getting stuck in an infinite loop checking for recursion
	// Note that we check for recursion to control the frequency of adding recurrent links rather than to prevent
	// any particular kind of error
	nodes_len := len(g.Nodes)
	thresh := nodes_len * nodes_len

	// Decide whether to make link recurrent
	do_recur := false
	if rand.Float64() < context.RecurOnlyProb {
		do_recur = true
	}

	// Find the first non-sensor so that the to-node won't look at sensors as possible destinations
	first_non_sensor := 0
	for _, n := range g.Nodes {
		if n.NType == network.SENSOR {
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
		bypass := false
		if node_2.NType == network.SENSOR {
			// Don't allow SENSORS to get input
			bypass = true
		} else {
			for _, gene := range g.Genes {
				if gene.Link.InNode.Id == node_1.Id &&
					gene.Link.OutNode.Id == node_2.Id &&
					gene.Link.IsRecurrent == do_recur {
					// link already exists
					bypass = true;
					break;
				}

			}
		}

		if !bypass {
			// check if link is open
			count := 0
			recur_flag := g.Phenotype.IsRecurrent(node_1.Analogue, node_2.Analogue, &count, thresh)

			// Exit if the network is faulty (contains an infinite loop)
			if count > thresh {
				neat.DebugLog(fmt.Sprintf("Recurency -> node in: %s <-> node out: %s",
					node_1.Analogue, node_2.Analogue))
				return false, errors.New("GENOME: ERROR: LOOP DETECTED DURING A RECURRENCY CHECK")
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
			if inn.InnovationType == NEWLINK &&
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
			new_weight := float64(neat.RandPosNeg()) * rand.Float64() * 10.0
			// read curr innovation with post increment
			curr_innov := pop.getInnovationNumberAndIncrement()

			// Create the new gene
			new_gene = NewGeneWithTrait(g.Traits[trait_num], new_weight, node_1, node_2,
				do_recur, curr_innov, new_weight)

			// Add the innovation
			new_innov := NewInnovationForRecurrentLink(node_1.Id, node_2.Id, curr_innov,
				new_weight, trait_num, do_recur)
			pop.Innovations = append(pop.Innovations, new_innov)
		} else if g.hasGene(new_gene) {
			// The gene for already occurred innovation already in this genome.
			// This may happen as result of parent genome mutation in current epoch which is
			// repeated in the child after parent's genome transferred to child during mating
			neat.DebugLog(
				fmt.Sprintf("GENOME: ALERT: Mutate add link innovation found [%t] in the same genome [%d] for gene: %s\n%s",
					innovation_found, g.Id, new_gene, g))
			return false, nil
		}

		// sanity check
		if new_gene.Link.InNode.Id == new_gene.Link.OutNode.Id && !do_recur {
			neat.DebugLog(fmt.Sprintf("Recurent link created when recurency is not enabled: %s", new_gene))
			return false, errors.New(fmt.Sprintf("GENOME: ERROR: Wrong gene created!\n%s", g))
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
			if gn.IsEnabled && gn.Link.InNode.NType != network.BIAS && rand.Float32() >= 0.3 {
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
			if gene.IsEnabled && gene.Link.InNode.NType != network.BIAS {
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
		if inn.InnovationType == NEWNODE &&
			inn.InNodeId == in_node.Id &&
			inn.OutNodeId == out_node.Id &&
			inn.OldInnovNum == gene.InnovationNum {

			// Create the new NNode
			new_node = network.NewNNodeInPlace(network.NEURON, inn.NewNodeId, network.HIDDEN)
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
		curr_node_id := pop.getCurrentNodeIdAndIncrement()

		// Create the new NNode
		new_node = network.NewNNodeInPlace(network.NEURON, curr_node_id, network.HIDDEN)
		// By convention, it will point to the first trait
		new_node.Trait = g.Traits[0]

		// get the current gene 1 innovation with post increment
		gene_innov_1 := pop.getInnovationNumberAndIncrement()
		// create gene with the current gene innovation
		new_gene_1 = NewGeneWithTrait(trait, 1.0, in_node, new_node, link.IsRecurrent, gene_innov_1, 0);

		// get the current gene 2 innovation with post increment
		gene_innov_2 := pop.getInnovationNumberAndIncrement()
		// create the second gene with this innovation incremented
		new_gene_2 = NewGeneWithTrait(trait, old_weight, new_node, out_node, false, gene_innov_2, 0);

		// Store innovation
		innov := NewInnovationForNode(in_node.Id, out_node.Id, gene_innov_1, gene_innov_2, new_node.Id, gene.InnovationNum)
		pop.Innovations = append(pop.Innovations, innov)
	} else if g.hasNode(new_node) {
		// The same add node innovation occurred in the same genome (parent) - just skip.
		// This may happen when parent of this organism experienced the same mutation in current epoch earlier
		// and after that parent's genome was duplicated to child by mating and the same mutation parameters
		// was selected again (in_node.Id, out_node.Id, gene.InnovationNum). As result the innovation with given
		// parameters will be found and new node will be created with ID which alredy exists in child genome.
		// If proceed than we will have duplicated Node and genes - so we're skipping this.
		neat.DebugLog(
			fmt.Sprintf("GENOME: ALERT: Add node innovation found [%t] in the same genome [%d] for node [%d]\n%s",
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
func (g *Genome) mutateLinkWeights(power, rate float64, mutation_type int) (bool, error) {
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

		rand_val := float64(neat.RandPosNeg()) * rand.Float64() * power
		if mutation_type == GAUSSIAN {
			rand_choice := rand.Float64()
			if rand_choice > gauss_point {
				gene.Link.Weight += rand_val
			} else if rand_choice > cold_gauss_point {
				gene.Link.Weight = rand_val
			}
		} else if mutation_type == COLD_GAUSSIAN {
			gene.Link.Weight = rand_val
		}

		// Record the innovation
		gene.MutationNum = gene.Link.Weight

		num += 1.0
	}

	return true, nil
}

// Perturb params in one trait
func (g *Genome) mutateRandomTrait(conf *neat.NeatContext) (bool, error) {
	if len(g.Traits) == 0 {
		return false, errors.New("Genome has no traits")
	}
	// Choose a random trait number
	trait_num := rand.Intn(len(g.Traits))

	// Retrieve the trait and mutate it
	g.Traits[trait_num].Mutate(conf.TraitMutationPower, conf.TraitParamMutProb)

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
func (g *Genome) mutateAllNonstructural(conf *neat.NeatContext) (bool, error) {
	res := false
	var err error
	if rand.Float64() < conf.MutateRandomTraitProb {
		// mutate random trait
		res, err = g.mutateRandomTrait(conf)
	}

	if err == nil && rand.Float64() < conf.MutateLinkTraitProb {
		// mutate link trait
		res, err = g.mutateLinkTrait(1)
	}

	if err == nil && rand.Float64() < conf.MutateNodeTraitProb {
		// mutate node trait
		res, err = g.mutateNodeTrait(1)
	}

	if err == nil && rand.Float64() < conf.MutateLinkWeightsProb {
		// mutate link weight
		res, err = g.mutateLinkWeights(conf.WeightMutPower, 1.0, GAUSSIAN)
	}

	if err == nil && rand.Float64() < conf.MutateToggleEnableProb {
		// mutate toggle enable
		res, err = g.mutateToggleEnable(1)
	}

	if err == nil && rand.Float64() < conf.MutateGeneReenableProb {
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
	new_traits := make([]*neat.Trait, len(gen.Traits))
	for i, tr := range gen.Traits {
		new_traits[i] = neat.NewTraitAvrg(tr, og.Traits[i])
	}

	// The new genes and nodes created
	new_genes := make([]*Gene, 0)
	new_nodes := make([]*network.NNode, 0)

	// NEW: Make sure all sensors and outputs are included (in case some inputs are disconnected)
	for _, curr_node := range og.Nodes {
		if curr_node.GenNodeLabel == network.INPUT ||
			curr_node.GenNodeLabel == network.BIAS ||
			curr_node.GenNodeLabel == network.OUTPUT {
			node_trait_num := 0
			if curr_node.Trait != nil {
				node_trait_num = curr_node.Trait.Id - gen.Traits[0].Id
			}
			// Create a new node off the sensor or output
			new_onode := network.NewNNodeCopy(curr_node, new_traits[node_trait_num])

			// Add the new node
			nodeInsert(new_nodes, new_onode)
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
			if new_gene.Link.InNode.Id == chosen_gene.Link.InNode.Id &&
				new_gene.Link.OutNode.Id == chosen_gene.Link.OutNode.Id &&
				new_gene.Link.IsRecurrent == chosen_gene.Link.IsRecurrent {
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
	new_genome := NewGenome(genomeid, new_traits, new_nodes, new_genes)

	//Return the baby Genome
	return new_genome, nil
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
	new_traits := make([]*neat.Trait, len(gen.Traits))
	for i, tr := range gen.Traits {
		new_traits[i] = neat.NewTraitAvrg(tr, og.Traits[i]) // construct by averaging
	}

	// The new genes and nodes created
	new_genes := make([]*Gene, 0)
	new_nodes := make([]*network.NNode, 0)

	// NEW: Make sure all sensors and outputs are included (in case some inputs are disconnected)
	for _, curr_node := range og.Nodes {
		if curr_node.GenNodeLabel == network.INPUT ||
			curr_node.GenNodeLabel == network.BIAS ||
			curr_node.GenNodeLabel == network.OUTPUT {
			node_trait_num := 0
			if curr_node.Trait != nil {
				node_trait_num = curr_node.Trait.Id - gen.Traits[0].Id
			}
			// Create a new node off the sensor or output
			new_onode := network.NewNNodeCopy(curr_node, new_traits[node_trait_num])

			// Add the new node
			nodeInsert(new_nodes, new_onode)
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
			if new_gene.Link.InNode.Id == chosen_gene.Link.InNode.Id &&
				new_gene.Link.OutNode.Id == chosen_gene.Link.OutNode.Id &&
				new_gene.Link.IsRecurrent == chosen_gene.Link.IsRecurrent {
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
	new_genome := NewGenome(genomeid, new_traits, new_nodes, new_genes)

	//Return the baby Genome
	return new_genome, nil
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
	new_traits := make([]*neat.Trait, len(gen.Traits))
	for i, tr := range gen.Traits {
		new_traits[i] = neat.NewTraitAvrg(tr, og.Traits[i]) // construct by averaging
	}

	// The new genes and nodes created
	new_genes := make([]*Gene, 0)
	new_nodes := make([]*network.NNode, 0)

	// NEW: Make sure all sensors and outputs are included (in case some inputs are disconnected)
	for _, curr_node := range og.Nodes {
		if curr_node.GenNodeLabel == network.INPUT ||
			curr_node.GenNodeLabel == network.BIAS ||
			curr_node.GenNodeLabel == network.OUTPUT {
			node_trait_num := 0
			if curr_node.Trait != nil {
				node_trait_num = curr_node.Trait.Id - gen.Traits[0].Id
			}
			// Create a new node off the sensor or output
			new_onode := network.NewNNodeCopy(curr_node, new_traits[node_trait_num])

			// Add the new node
			nodeInsert(new_nodes, new_onode)
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
			if new_gene.Link.InNode.Id == chosen_gene.Link.InNode.Id &&
				new_gene.Link.OutNode.Id == chosen_gene.Link.OutNode.Id &&
				new_gene.Link.IsRecurrent == chosen_gene.Link.IsRecurrent {
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
	new_genome := NewGenome(genomeid, new_traits, new_nodes, new_genes)

	//Return the baby Genome
	return new_genome, nil
}

/* ******** COMPATIBILITY CHECKING METHODS * ********/

// This function gives a measure of compatibility between two Genomes by computing a linear combination of 3
// characterizing variables of their compatibility. The 3 variables represent PERCENT DISJOINT GENES,
// PERCENT EXCESS GENES, MUTATIONAL DIFFERENCE WITHIN MATCHING GENES. So the formula for compatibility
// is:  disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg
// The 3 coefficients are global system parameters */
func (g *Genome) compatibility(og *Genome, conf *neat.NeatContext) float64 {
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
	comp := conf.DisjointCoeff * num_disjoint + conf.ExcessCoeff * num_excess +
		conf.MutdiffCoeff * (mut_diff_total / num_matching)

	return comp
}




