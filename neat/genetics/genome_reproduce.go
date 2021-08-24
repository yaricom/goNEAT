package genetics

import (
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"math/rand"
)

/* ****** MATING METHODS ***** */

// This method mates this Genome with another Genome g. For every point in each Genome, where each Genome shares
// the innovation number, the Gene is chosen randomly from either parent.  If one parent has an innovation absent in
// the other, the baby may inherit the innovation if it is from the more fit parent.
// The new Genome is given the id in the genomeId argument.
func (g *Genome) mateMultipoint(og *Genome, genomeId int, fitness1, fitness2 float64) (*Genome, error) {
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
				avgGene.Link.ConnectionWeight = (p1gene.Link.ConnectionWeight + p2gene.Link.ConnectionWeight) / 2.0 // WEIGHTS AVERAGED HERE

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
func (g *Genome) mateSinglePoint(og *Genome, genomeId int) (*Genome, error) {
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
					avgGene.Link.ConnectionWeight = (p1gene.Link.ConnectionWeight + p2gene.Link.ConnectionWeight) / 2.0 // WEIGHTS AVERAGED HERE

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
			// no gene was chosen - no need to process further - exit cycle
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
