package genetics

import (
	"errors"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"math/rand"
)

/* ******* MUTATORS ******* */

// Mutate the genome by adding connections to disconnected sensors (input, bias type neurons).
// The reason this mutator is important is that if we can start NEAT with some inputs disconnected,
// then we can allow NEAT to decide which inputs are important.
// This process has two good effects:
// 	(1) You can start minimally even in problems with many inputs and
// 	(2) you don't need to know a priori what the important features of the domain are.
// If all sensors already connected than do nothing.
func (g *Genome) mutateConnectSensors(innovations InnovationsObserver, _ *neat.Options) (bool, error) {

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
			for _, inn := range innovations.Innovations() {
				if inn.innovationType == newLinkInnType &&
					inn.InNodeId == sensor.Id &&
					inn.OutNodeId == output.Id &&
					!inn.IsRecurrent {

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
				newWeight := float64(math.RandSign()) * rand.Float64() * 10.0
				// read next innovation id
				nextInnovId := innovations.NextInnovationNumber()

				// Create the new gene
				gene = NewGeneWithTrait(g.Traits[traitNum], newWeight, sensor, output,
					false, nextInnovId, newWeight)

				// Add the innovation for created link
				newInnovation := NewInnovationForLink(sensor.Id, output.Id, nextInnovId,
					newWeight, traitNum)
				innovations.StoreInnovation(*newInnovation)
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
func (g *Genome) mutateAddLink(innovations InnovationsObserver, opts *neat.Options) (bool, error) {
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
	if rand.Float64() < opts.RecurOnlyProb {
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
	for tryCount < opts.NewLinkTries {
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
				tryCount = opts.NewLinkTries
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
		for _, inn := range innovations.Innovations() {
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
			newWeight := float64(math.RandSign()) * rand.Float64() * 10.0
			// read next innovation id
			nextInnovId := innovations.NextInnovationNumber()

			// Create the new gene
			gene = NewGeneWithTrait(g.Traits[traitNum], newWeight, node1, node2,
				doRecur, nextInnovId, newWeight)

			// Add the innovation
			innovation := NewInnovationForRecurrentLink(node1.Id, node2.Id, nextInnovId,
				newWeight, traitNum, doRecur)
			innovations.StoreInnovation(*innovation)
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
func (g *Genome) mutateAddNode(innovations InnovationsObserver, nodeIdGenerator network.NodeIdGenerator, opts *neat.Options) (bool, error) {
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
	oldWeight := link.ConnectionWeight
	// Get the old link's trait
	trait := link.Trait

	// Extract the nodes
	inNode, outNode := link.InNode, link.OutNode
	if inNode == nil || outNode == nil {
		return false, fmt.Errorf("mutateAddNode: Anomalous link found with either IN or OUT node not set. %s", link)
	}

	var gene1, gene2 *Gene
	var node *network.NNode

	// Check to see if this innovation already occurred in the population
	innovationFound := false
	for _, inn := range innovations.Innovations() {
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
		newNodeId := nodeIdGenerator.NextNodeId()

		// Create the new NNode
		node = network.NewNNode(newNodeId, network.HiddenNeuron)
		// By convention, it will point to the first trait
		node.Trait = g.Traits[0]
		// Set node activation function as random from a list of types registered with opts
		if activationType, err := opts.RandomNodeActivationType(); err != nil {
			return false, err
		} else {
			node.ActivationType = activationType
		}

		// get the next innovation id for gene 1
		gene1Innovation := innovations.NextInnovationNumber()
		// create gene with the current gene innovation
		gene1 = NewGeneWithTrait(trait, 1.0, inNode, node, link.IsRecurrent, gene1Innovation, 0)

		// get the next innovation id for gene 2
		gene2Innovation := innovations.NextInnovationNumber()
		// create the second gene with this innovation incremented
		gene2 = NewGeneWithTrait(trait, oldWeight, node, outNode, false, gene2Innovation, 0)

		// Store innovation
		innovation := NewInnovationForNode(inNode.Id, outNode.Id, gene1Innovation, gene2Innovation, node.Id, gene.InnovationNum)
		innovations.StoreInnovation(*innovation)
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

		random := float64(math.RandSign()) * rand.Float64() * power
		if mutationType == gaussianMutator {
			randChoice := rand.Float64()
			if randChoice > gaussPoint {
				gene.Link.ConnectionWeight += random
			} else if randChoice > coldGaussPoint {
				gene.Link.ConnectionWeight = random
			}
		} else if mutationType == goldGaussianMutator {
			gene.Link.ConnectionWeight = random
		}

		// Record the innovation
		gene.MutationNum = gene.Link.ConnectionWeight

		num += 1.0
	}

	return true, nil
}

// Perturb params in one trait
func (g *Genome) mutateRandomTrait(context *neat.Options) (bool, error) {
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
func (g *Genome) mutateGeneReEnable() (bool, error) {
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
func (g *Genome) mutateAllNonstructural(context *neat.Options) (bool, error) {
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
		res, err = g.mutateGeneReEnable()
	}
	return res, err
}
