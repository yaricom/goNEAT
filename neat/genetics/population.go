package genetics

import (
	"math/rand"
	"errors"
	"github.com/yaricom/goNEAT/neat"

	"io"
	"fmt"
	"sort"
	"bufio"
	"strings"
	"bytes"
	"strconv"
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
func NewPopulation(g *Genome, context *neat.NeatContext) (*Population, error) {
	if context.PopSize <= 0 {
		return nil, errors.New(
			fmt.Sprintf("Wrong population size in the context: %d", context.PopSize))
	}

	pop := newPopulation()
	err := pop.spawn(g, context)
	if err != nil {
		return nil, err
	}
	return pop, nil
}

// Special constructor to create a population of random topologies uses
// NewGenomeRand(new_id, in, out, n, nmax int, recurrent bool, link_prob float64)
// See the Genome constructor above for the argument specifications
func NewPopulationRandom(in, out, nmax int, recurrent bool, link_prob float64, context *neat.NeatContext) (*Population, error) {
	if context.PopSize <= 0 {
		return nil, errors.New(
			fmt.Sprintf("Wrong population size in the context: %d", context.PopSize))
	}

	pop := newPopulation()
	for count := 0; count < context.PopSize; count++ {
		gen := NewGenomeRand(count, in, out, rand.Intn(nmax), nmax, recurrent, link_prob)
		pop.Organisms = append(pop.Organisms, NewOrganism(0.0, gen, 1))
	}
	pop.currNodeId = in + out + nmax + 1
	pop.currInnovNum = int64((in + out + nmax) * (in + out + nmax) + 1)

	err := pop.speciate(context)
	if err != nil {
		return nil, err
	}

	return pop, nil
}

// Reads population from provided reader
func ReadPopulation(ir io.Reader, context *neat.NeatContext) (pop *Population, err error) {
	pop = newPopulation()

	// Loop until file is finished, parsing each line
	scanner := bufio.NewScanner(ir)
	scanner.Split(bufio.ScanLines)
	var out_buff *bytes.Buffer
	var id_check int
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, " ", 2)
		if len(parts) < 2 {
			return nil, errors.New(fmt.Sprintf("Line: [%s] can not be split when reading Population", line))
		}
		switch parts[0] {
		case "genomestart":
			out_buff = bytes.NewBufferString(fmt.Sprintf("genomestart %s", parts[1]))
			id_check, err = strconv.Atoi(parts[1])
			if err != nil {
				return nil, err
			}
		case "genomeend":
			fmt.Fprintf(out_buff, "genomeend %d", id_check)
			new_genome, err := ReadGenome(bufio.NewReader(out_buff), id_check)
			if err != nil {
				return nil, err
			}
			// add new organism for read genome
			new_organism := NewOrganism(0.0, new_genome, 1)
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
			// clear buffer
			out_buff = nil
			id_check = -1

		case "/*":
			// read all comments and print it
			neat.InfoLog(line)
		default:
			// write line to buffer
			fmt.Fprintln(out_buff, line)
		}

	}
	err = pop.speciate(context)
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

// Writes given population by species
func (p *Population) WriteBySpecies(w io.Writer) {
	// Step through the Species and write them
	for _, sp := range p.Species {
		sp.Write(w)
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
func (p *Population) spawn(g *Genome, context *neat.NeatContext) error {
	var new_genome *Genome
	for count := 0; count < context.PopSize; count++ {
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
	err = p.speciate(context)

	return err
}

// Speciate separates the organisms into species by checking compatibilities against a threshold.
// Any organism that does is not compatible with the first organism in any existing species becomes a new species.
func (p *Population) speciate(context *neat.NeatContext) error {
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
					curr_org.GNome.compatibility(comp_org.GNome, context) < context.CompatThreshold {
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
func (p *Population) Verify() (bool, error) {
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
func (p *Population) Epoch(generation int, context *neat.NeatContext) (bool, error) {
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
	neat.DebugLog(fmt.Sprintf(
		"POPULATION: Generation %d: overall average fitness = %.3f, # of organisms: %d, # of species: %d\n",
		generation, overall_average, len(p.Organisms), len(p.Species)))

	// Now compute expected number of offspring for each individual organism
	if overall_average != 0 {
		for _, o := range p.Organisms {
			o.ExpectedOffspring = o.Fitness / overall_average
		}
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
			neat.DebugLog(
				fmt.Sprintf("POPULATION: Population died !!! (expected/total) %d/%d",
					final_expected, total_organisms))
			for _, sp := range p.Species {
				sp.ExpectedOffspring = 0
			}
			best_species.ExpectedOffspring = total_organisms
		}
	}

	// Remove stagnated species which can not produce any offspring any more
	species_to_keep := make([]*Species, 0)
	for _, sp := range p.Species {
		if sp.ExpectedOffspring > 0 {
			species_to_keep = append(species_to_keep, sp)
		}
	}
	p.Species = species_to_keep

	// Stick the Species pointers into a new Species list for sorting
	sorted_species := make([]*Species, len(p.Species))
	copy(sorted_species, p.Species)

	// Sort the Species by max original fitness of its first organism
	sort.Sort(sort.Reverse(ByOrganismOrigFitness(sorted_species)))

	// Used in debugging to see why (if) best species dies
	best_species_id := sorted_species[0].Id
	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog("POPULATION: >> Sorted Species START <<")
		for _, sp := range sorted_species {
			// Print out for Debugging/viewing what's going on
			neat.DebugLog(
				fmt.Sprintf("POPULATION: >> Orig. fitness of Species %d (Size %d): %f, current fitness: %f, expected offspring: %d, last improved %d \n",
				sp.Id, len(sp.Organisms), sp.Organisms[0].OriginalFitness, sp.Organisms[0].Fitness, sp.ExpectedOffspring, (sp.Age - sp.AgeOfLastImprovement)))
		}
		neat.DebugLog("POPULATION: >> Sorted Species END <<\n")

	}

	// Check for Population-level stagnation
	curr_species := sorted_species[0]
	curr_species.Organisms[0].IsPopulationChampion = true // DEBUG marker of the best of pop
	if curr_species.Organisms[0].OriginalFitness > p.HighestFitness {
		p.HighestFitness = curr_species.Organisms[0].OriginalFitness
		p.HighestLastChanged = 0
		neat.DebugLog(fmt.Sprintf("POPULATION: NEW POPULATION RECORD FITNESS: %f of SPECIES with ID: %d\n", p.HighestFitness, best_species_id))

	} else {
		p.HighestLastChanged += 1
		neat.DebugLog(fmt.Sprintf(" generations since last population fitness record: %f\n", p.HighestFitness))
	}

	// Check for stagnation - if there is stagnation, perform delta-coding
	if p.HighestLastChanged >= context.DropOffAge + 5 {
		neat.DebugLog("POPULATION: PERFORMING DELTA CODING")

		p.HighestLastChanged = 0
		half_pop := context.PopSize / 2

		neat.DebugLog(fmt.Sprintf("half_pop: [%d] (pop_size - halfpop): [%d]\n",
			half_pop, context.PopSize - half_pop))

		if len(sorted_species) > 1 {
			// Assign population to first two species
			curr_species.Organisms[0].SuperChampOffspring = half_pop
			curr_species.ExpectedOffspring = half_pop
			curr_species.AgeOfLastImprovement = curr_species.Age

			// process the second species
			curr_species = sorted_species[1]
			// NOTE: PopSize can be odd. That's why we use subtraction below
			curr_species.Organisms[0].SuperChampOffspring = context.PopSize - half_pop
			curr_species.ExpectedOffspring = context.PopSize - half_pop
			curr_species.AgeOfLastImprovement = curr_species.Age

			// Get rid of all species after the first two
			for i := 2; i < len(sorted_species); i++ {
				sorted_species[i].ExpectedOffspring = 0
			}
		} else {
			curr_species = sorted_species[0]
			curr_species.Organisms[0].SuperChampOffspring = context.PopSize
			curr_species.ExpectedOffspring = context.PopSize
			curr_species.AgeOfLastImprovement = curr_species.Age
		}
	} else if context.BabiesStolen > 0 {
		// STOLEN BABIES: The system can take expected offspring away from worse species and give them
		// to superior species depending on the system parameter BabiesStolen (when BabiesStolen > 0)
		stolen_babies := 0 // Babies taken from the bad species and given to the champs

		// Take away a constant number of expected offspring from the worst few species
		for i := len(sorted_species) - 1; i >= 0 && stolen_babies < context.BabiesStolen; i-- {
			curr_species = sorted_species[i]
			if curr_species.Age > 5 && curr_species.ExpectedOffspring > 2 {
				if curr_species.ExpectedOffspring - 1 >= context.BabiesStolen - stolen_babies {
					// This species has enough to finish off the stolen pool
					curr_species.ExpectedOffspring -= context.BabiesStolen - stolen_babies
					stolen_babies = context.BabiesStolen
				} else {
					// Not enough here to complete the pool of stolen
					stolen_babies += curr_species.ExpectedOffspring - 1
					curr_species.ExpectedOffspring = 1
				}
			}
		}

		neat.DebugLog(fmt.Sprintf("POPULATION: STOLEN BABIES: %d\n", stolen_babies))

		// Mark the best champions of the top species to be the super champs who will take on the extra
		// offspring for cloning or mutant cloning.
		// Determine the exact number that will be given to the top three.
		// They will get, in order, 1/5 1/5 and 1/10 of the stolen babies
		stolen_blocks := []int{context.BabiesStolen / 5, context.BabiesStolen / 5, context.BabiesStolen / 10}
		block_index := 0
		for _, curr_species = range sorted_species {
			if curr_species.lastImproved() > context.DropOffAge {
				// Don't give a chance to dying species even if they are champs
				continue
			}

			if block_index < 3 && stolen_babies >= stolen_blocks[block_index] {
				// Give stolen babies to the top three in 1/5 1/5 and 1/10 ratios
				curr_species.Organisms[0].SuperChampOffspring = stolen_blocks[block_index]
				curr_species.ExpectedOffspring += stolen_blocks[block_index]
				stolen_babies -= stolen_blocks[block_index]
			} else if block_index >= 3 {
				// Give stolen to the rest in random ratios
				if rand.Float64() > 0.1 {
					// Randomize a little which species get boosted by a super champ
					if stolen_babies > 3 {
						curr_species.Organisms[0].SuperChampOffspring = 3
						curr_species.ExpectedOffspring += 3
						stolen_babies -= 3
					} else {
						curr_species.Organisms[0].SuperChampOffspring = stolen_babies
						curr_species.ExpectedOffspring += stolen_babies
						stolen_babies = 0
					}
				}
			}

			if stolen_babies <= 0 {
				break
			}
			block_index++
		}
		// If any stolen babies aren't taken, give them to species #1's champ
		if stolen_babies > 0 {
			curr_species = sorted_species[0]
			curr_species.Organisms[0].SuperChampOffspring += stolen_babies
			curr_species.ExpectedOffspring += stolen_babies
		}
	}

	// Kill off all Organisms marked for death. The remainder will be allowed to reproduce.
	org_to_keep := make([]*Organism, 0)
	for _, curr_org := range p.Organisms {
		if curr_org.ToEliminate {
			// Remove the organism from its Species
			_, err := curr_org.SpeciesOf.removeOrganism(curr_org)
			if err != nil {
				return false, err
			}
		} else {
			// Keep organism in population
			org_to_keep = append(org_to_keep, curr_org)
		}
	}
	// Keep only remained organisms in the population
	p.Organisms = org_to_keep

	neat.DebugLog("POPULATION: Start Reproduction >>>>>")

	// Perform reproduction. Reproduction is done on a per-Species basis
	best_species_reproduced := false
	// TODO (So this could be parallelised potentially)
	for _, curr_species := range p.Species {
		reproduced, err := curr_species.reproduce(generation, p, sorted_species, context)
		if err != nil {
			return false, err
		}
		if curr_species.Id == best_species_id {
			// store flag if best species reproduced - it will be used to determine if best species
			// produced offspring before died
			best_species_reproduced = reproduced
		}
	}

	neat.DebugLog("POPULATION: >>>>> Reproduction Complete")

	// Destroy and remove the old generation from the organisms and species
	for _, curr_org := range p.Organisms {
		// Remove the organism from its Species
		_, err := curr_org.SpeciesOf.removeOrganism(curr_org)
		if err != nil {
			return false, err
		}

		if neat.LogLevel == neat.LogLevelDebug && curr_org.SpeciesOf.Id == best_species_id {
			neat.DebugLog(fmt.Sprintf("POPULATION: Removed organism [%d] from best species [%d] - %d organisms remained",
			curr_org.GNome.Id, best_species_id, len(curr_org.SpeciesOf.Organisms)))
		}
	}
	p.Organisms = make([]*Organism, 0)

	// Remove all empty Species and age ones that survive.
	// As this happens, create master organism list for the new generation.
	org_count := 0
	species_to_keep = make([]*Species, 0)
	for _, curr_species := range p.Species {
		if len(curr_species.Organisms) > 0 {
			// Age surviving Species
			if curr_species.IsNovel {
				curr_species.IsNovel = false
			} else {
				curr_species.Age += 1
			}
			// Rebuild master Organism list of population: NUMBER THEM as they are added to the list
			for _, curr_org := range curr_species.Organisms {
				curr_org.GNome.Id = org_count
				p.Organisms = append(p.Organisms, curr_org)
				org_count++
			}
			// keep this species
			species_to_keep = append(species_to_keep, curr_species)
		} else if neat.LogLevel == neat.LogLevelDebug {
			neat.DebugLog(fmt.Sprintf("POPULATION: >> Species [%d] have not survived reproduction!",
				curr_species.Id))
		}
	}
	// Keep only survived species
	p.Species = species_to_keep

	neat.DebugLog(fmt.Sprintf("POPULATION: # of species survived: %d, # of organisms survived: %d\n",
		len(p.Species), len(p.Organisms)))

	// Remove the innovations of the current generation
	p.Innovations = make([]*Innovation, 0)

	// Check to see if the best species died somehow. We don't want this to happen!!!
	// N.B. the mutated offspring of best species may be added to other more compatible species and as result
	// the best species from previous generation will be removed, but their offspring still be alive
	best_ok := false
	for _, curr_species := range p.Species {
		neat.DebugLog(fmt.Sprintf("POPULATION: %d <> %d\n", curr_species.Id, best_species_id))
		if curr_species.Id == best_species_id {
			best_ok = true
			break
		}
	}
	if !best_ok && !best_species_reproduced{
		return false, errors.New("POPULATION: The best species died without offspring!")
	} else {
		neat.DebugLog(fmt.Sprintf("POPULATION: The best survived species Id: %d", best_species_id))
	}

	// DEBUG: Checking the top organism's duplicate in the next gen
	// This prints the champ's child to the screen
	if neat.LogLevel == neat.LogLevelDebug {
		for _, curr_org := range p.Organisms {
			if curr_org.IsPopulationChampionChild {
				neat.DebugLog(
					fmt.Sprintf("POPULATION: At end of reproduction cycle, the child of the pop champ is: %s",
						curr_org.GNome))
			}
		}
	}
	neat.DebugLog(fmt.Sprintf("POPULATION: >>>>> Epoch %d complete\n", generation))

	return true, nil
}
