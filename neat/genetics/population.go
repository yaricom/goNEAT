package genetics

import (
	"math/rand"
	"errors"
	"github.com/yaricom/goNEAT/neat"

	"io"
	"fmt"
	"sort"
)

// A Population is a group of Organisms including their species
type Population struct {
	// Species in the Population. Note that the species should comprise all the genomes
	Species            []*Species
	// The organisms in the Population
	Organisms          []*Organism
	// The highest species number
	LastSpecies        int
	// For holding the genetic innovations of the newest generation
	Innovations        []*Innovation
	// An integer that when above zero tells when the first winner appeared
	WinnerGen          int
	// The last generation played
	FinalGen           int

	// Stagnation detector
	HighestFitness     float64
	// If too high, leads to delta coding
	HighestLastChanged int

	/* Fitness Statistics */
	MeanFitness        float64
	Variance           float64
	StandardDev        float64


	// The current innovation number for population
	currInnovNum       int64
	// The current ID for new node in population
	currNodeId         int
}

// Construct off of a single spawning Genome
func NewPopulation(g *Genome, size int, conf *neat.NeatContext) (*Population, error) {
	pop := newPopulation()

	err := pop.spawn(g, size, conf)
	if err != nil {
		return nil, err
	}
	return pop, nil
}

// Special constructor to create a population of random topologies uses
// NewGenomeRand(new_id, in, out, n, nmax int, recurrent bool, link_prob float64)
// See the Genome constructor above for the argument specifications
func NewPopulationRandom(size, in, out, nmax int, recurrent bool, link_prob float64, conf *neat.NeatContext) (*Population, error) {
	pop := newPopulation()

	for count := 0; count < size; count++ {
		gen := NewGenomeRand(count, in, out, rand.Intn(nmax), nmax, recurrent, link_prob)
		pop.Organisms = append(pop.Organisms, NewOrganism(0, gen, 1))
	}
	pop.currNodeId = in + out + nmax + 1
	pop.currInnovNum = int64((in + out + nmax) * (in + out + nmax) + 1)

	err := pop.speciate(conf)
	if err != nil {
		return nil, err
	}

	return pop, nil
}

// Reads population from provided reader
func ReadPopulation(r io.Reader, conf *neat.NeatContext) (*Population, error) {
	pop := newPopulation()

	var cur_word string
	for true {
		_, ioerr := fmt.Fscanf(r, "%s", &cur_word)
		if ioerr == io.EOF {
			break
		}
		if cur_word == "genomestart" {
			var id_check int
			fmt.Fscanf(r, "%d", &id_check)
			new_genome, err := ReadGenome(r, id_check)
			if err != nil {
				return nil, err
			}
			// add new organism for read genome
			new_organism := NewOrganism(0, new_genome, 1)
			pop.Organisms = append(pop.Organisms, new_organism)

			if last_node_id, err := new_genome.getLastNodeId(); err == nil {
				if pop.currNodeId < last_node_id {
					pop.currNodeId = last_node_id
				}
			} else {
				return nil, err
			}

			if last_gene_innov_num, err := new_genome.getLastGeneInnovNum(); err == nil {
				if pop.currInnovNum < last_gene_innov_num {
					pop.currInnovNum = last_gene_innov_num
				}
			} else {
				return nil, err
			}

		} else if cur_word == "/*" {
			// read comments and print it
			for cur_word != "*/" {
				fmt.Fscanf(r, "%s", &cur_word)
				fmt.Println(cur_word)
			}
		}
	}
	err := pop.speciate(conf)
	if err != nil {
		return nil, err
	} else {
		return pop, nil
	}
}

// Writes given population to a writer
func (p *Population) Write(w io.Writer) {
	// Prints all the Organisms' Genomes to the outFile
	for _, o := range p.Organisms {
		o.GNome.Write(w)
	}
}

// Default private constructor
func newPopulation() *Population {
	return &Population{
		WinnerGen:0,
		HighestFitness:0.0,
		HighestLastChanged:0,
		Species:make([]*Species, 0),
		Organisms:make([]*Organism, 0),
		Innovations:make([]*Innovation, 0),
	}
}

// Returns current innovation number and increment innovations number counter after that
func (p *Population) getInnovationNumberAndIncrement() int64 {
	inn_num := p.currInnovNum
	p.currInnovNum++
	return inn_num
}
// Returns the current node ID which can be used to create new node in population and increment it after
func (p *Population) getCurrentNodeIdAndIncrement() int {
	node_id := p.currNodeId
	p.currNodeId++
	return node_id
}

// Create a population of size size off of Genome g. The new Population will have the same topology as g
// with link weights slightly perturbed from g's
func (p *Population) spawn(g *Genome, size int, conf *neat.NeatContext) error {
	var new_genome *Genome
	for count := 0; count < size; count++ {
		new_genome = g.duplicate(count)
		_, err := new_genome.mutateLinkWeights(1.0, 1.0, GAUSSIAN)
		if err != nil {
			return err
		}
		new_organism := NewOrganism(0.0, new_genome, 1)
		p.Organisms = append(p.Organisms, new_organism)
	}
	//Keep a record of the innovation and node number we are on
	var err error
	p.currNodeId, err = new_genome.getLastNodeId()
	p.currInnovNum, err = new_genome.getLastGeneInnovNum()

	if err != nil {
		return err
	}

	// Separate the new Population into species
	err = p.speciate(conf)

	return err
}

// Speciate separates the organisms into species by checking compatibilities against a threshold.
// Any organism that does is not compatible with the first organism in any existing species becomes a new species.
func (p *Population) speciate(conf *neat.NeatContext) error {
	if len(p.Organisms) == 0 {
		return errors.New("There is no organisms to speciate from")
	}

	// Species counter
	species_counter := 0
	// Step through all known organisms
	for _, curr_org := range p.Organisms {
		if len(p.Species) == 0 {
			// Create the first species
			new_species := NewSpecies(species_counter)
			p.Species = append(p.Species, new_species)
			new_species.addOrganism(curr_org)
			curr_org.SpeciesOf = new_species
			species_counter++
		} else {
			// For each organism, search for a species it is compatible to
			done := false
			for _, curr_species := range p.Species {
				comp_org := curr_species.firstOrganism()
				// compare current organism with first organism in current specie
				if comp_org != nil &&
					curr_org.GNome.compatibility(comp_org.GNome, conf) < conf.CompatThreshold {
					// Found compatible species, so add this organism to it
					curr_species.addOrganism(curr_org)
					curr_org.SpeciesOf = curr_species // Point organism to its species
					done = true
					break
				}
			}
			// If we didn't find a match, create a new species
			if !done {
				new_species := NewSpecies(species_counter)
				p.Species = append(p.Species, new_species)
				new_species.addOrganism(curr_org)
				curr_org.SpeciesOf = new_species
				species_counter++
			}
		}
	}
	p.LastSpecies = species_counter // Keep track of highest species

	return nil
}

// Run verify on all Genomes in this Population (Debugging)
func (p *Population) verify() (bool, error) {
	res := true
	var err error
	for _, o := range p.Organisms {
		res, err = o.GNome.verify()
		if err != nil {
			return false, err
		}
	}
	return res, nil
}

// Turnover the population to a new generation using fitness
// The generation argument is the next generation
func (p *Population) epoch(generation int, context *neat.NeatContext) bool {
	// Use Species' ages to modify the objective fitness of organisms in other words, make it more fair for younger
	// species so they have a chance to take hold and also penalize stagnant species. Then adjust the fitness using
	// the species size to "share" fitness within a species. Then, within each Species, mark for death those below
	// survival_thresh * average
	for _, sp := range p.Species {
		sp.adjustFitness(context)
	}

	// Used to compute average fitness over all Organisms
	total := 0.0
	total_organisms := len(p.Organisms)

	// Go through the organisms and add up their fitnesses to compute the overall average
	for _, o := range p.Organisms {
		total += o.Fitness
	}
	// The average modified fitness among ALL organisms
	overall_average := total / float64(total_organisms)
	context.DebugLog(fmt.Sprintf("Generation %d: overall average fitness = %f\n", generation, overall_average))

	// Now compute expected number of offspring for each individual organism
	for _, o := range p.Organisms {
		o.ExpectedOffspring = o.Fitness / overall_average
	}

	//The fractional parts of expected offspring that can be used only when they accumulate above 1 for the purposes
	// of counting Offspring
	skim := 0.0
	// precision checking
	total_expected := 0

	// Now add those offspring up within each Species to get the number of offspring per Species
	for _, sp := range p.Species {
		skim = sp.countOffspring(skim)
		total_expected += sp.ExpectedOffspring
	}

	// Need to make up for lost floating point precision in offspring assignment.
	// If we lost precision, give an extra baby to the best Species
	if total_expected < total_organisms {
		// Find the Species expecting the most
		var best_species *Species
		max_expected := 0
		final_expected := 0
		for _, sp := range p.Species {
			if sp.ExpectedOffspring >= max_expected {
				max_expected = sp.ExpectedOffspring
				best_species = sp
			}
			final_expected += sp.ExpectedOffspring
		}
		// Give the extra offspring to the best species
		best_species.ExpectedOffspring += 1
		final_expected++

		// If we still aren't at total, there is a problem. Note that this can happen if a stagnant Species
		// dominates the population and then gets killed off by its age. Then the whole population plummets in
		// fitness. If the average fitness is allowed to hit 0, then we no longer have an average we can use to
		// assign offspring.
		if final_expected < total_organisms {
			fmt.Println("!!! Population died !!!")
			for _, sp := range p.Species {
				sp.ExpectedOffspring = 0
			}
			best_species.ExpectedOffspring = total_organisms
		}
	}

	// Stick the Species pointers into a new Species list for sorting
	sorted_species := make([]*Species, len(p.Species))
	copy(sorted_species, p.Species)

	// Sort the Species by max original fitness of its first organism
	sort.Sort(ByOrganismOrigFitness(sorted_species))

	// Used in debugging to see why (if) best species dies
	// TODO best_species_num := sorted_species[0].Id
	if context.IsDebugEnabled {
		for _, sp := range sorted_species {
			// Print out for Debugging/viewing what's going on
			fmt.Printf("Orig. fitness of Species %d (Size %d): %f last improved %d \n",
				sp.Id, len(sp.Organisms), sp.Organisms[0].OriginalFitness, (sp.Age - sp.AgeOfLastImprovement))
		}
	}

	// Check for Population-level stagnation
	sorted_species[0].Organisms[0].IsPopulationChampion = true // DEBUG marker of the best of pop
	if sorted_species[0].Organisms[0].OriginalFitness > p.HighestFitness {
		p.HighestFitness = sorted_species[0].Organisms[0].OriginalFitness
		p.HighestLastChanged = 0
		context.DebugLog(fmt.Sprintf("NEW POPULATION RECORD FITNESS: %f\n", p.HighestFitness))

	} else {
		p.HighestLastChanged += 1
		context.DebugLog(fmt.Sprintf(" generations since last population fitness record: %f\n", p.HighestFitness))
	}

	// Check for stagnation- if there is stagnation, perform delta-coding
	if p.HighestLastChanged>= context.DropOffAge+5 {
		context.DebugLog("PERFORMING DELTA CODING")

		// TODO implement this

	}


	return false
}
