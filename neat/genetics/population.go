package genetics

import (
	"math/rand"
	"errors"
	"github.com/yaricom/goNEAT/neat"

	"io"
	"fmt"
	"bufio"
	"strings"
	"bytes"
	"strconv"
	"math"
	"sync/atomic"
	"sync"
)

// A Population is a group of Organisms including their species
type Population struct {
	// Species in the Population. Note that the species should comprise all the genomes
	Species                  []*Species
	// The organisms in the Population
	Organisms                []*Organism
	// The highest species number
	LastSpecies              int
	// For holding the genetic innovations of the newest generation
	Innovations              []*Innovation
	// An integer that when above zero tells when the first winner appeared
	WinnerGen                int
	// The last generation played
	FinalGen                 int

	// Stagnation detector
	HighestFitness           float64
	// The number of epochs when highest fitness was recorded for this population. If it was too long before
	// than delta coding will be applied to avoid population's fitness stagnation
	EpochsHighestLastChanged int

	/* Fitness Statistics */
	MeanFitness              float64
	Variance                 float64
	StandardDev              float64

	// The next innovation number for population
	nextInnovNum             int64
	// The next ID for new node in population
	nextNodeId               int32

	// The mutex to guard against concurrent modifications
	mutex                    *sync.Mutex
}

// The auxiliary data type to hold results of parallel reproduction sent over the wires
type reproductionResult struct {
	babies_stored int
	babies        []byte
	err           error
	species_id    int
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
		gen := newGenomeRand(count, in, out, rand.Intn(nmax), nmax, recurrent, link_prob)
		org, err := NewOrganism(0.0, gen, 1)
		if err != nil {
			return nil, err
		}
		pop.Organisms = append(pop.Organisms, org)
	}
	pop.nextNodeId = int32(in + out + nmax + 1)
	pop.nextInnovNum = int64((in + out + nmax) * (in + out + nmax) + 1)

	err := pop.speciate(pop.Organisms, context)
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
			new_organism, err := NewOrganism(0.0, new_genome, 1)
			if err != nil {
				return nil, err
			}
			pop.Organisms = append(pop.Organisms, new_organism)

			if last_node_id, err := new_genome.getLastNodeId(); err == nil {
				if pop.nextNodeId < int32(last_node_id) {
					pop.nextNodeId = int32(last_node_id + 1)
				}
			} else {
				return nil, err
			}

			if last_gene_innov_num, err := new_genome.getNextGeneInnovNum(); err == nil {
				if pop.nextInnovNum < last_gene_innov_num {
					pop.nextInnovNum = last_gene_innov_num
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
	err = pop.speciate(pop.Organisms, context)
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
		o.Genotype.Write(w)
	}
}

// Writes given population by species
func (p *Population) WriteBySpecies(w io.Writer) {
	// Step through the Species and write them
	for _, sp := range p.Species {
		sp.Write(w)
	}
}

// Run verify on all Genomes in this Population (Debugging)
func (p *Population) Verify() (bool, error) {
	res := true
	var err error
	for _, o := range p.Organisms {
		res, err = o.Genotype.verify()
		if err != nil {
			return false, err
		}
	}
	return res, nil
}

// Default private constructor
func newPopulation() *Population {
	return &Population{
		WinnerGen:0,
		HighestFitness:0.0,
		EpochsHighestLastChanged:0,
		Species:make([]*Species, 0),
		Organisms:make([]*Organism, 0),
		Innovations:make([]*Innovation, 0),
		mutex:&sync.Mutex{},
	}
}

// Returns current innovation number and increment innovations number counter after that
func (p *Population) getNextInnovationNumberAndIncrement() int64 {
	return atomic.AddInt64(&p.nextInnovNum, 1)
}
// Returns the next node ID which can be used to create new node in population
func (p *Population) getNextNodeIdAndIncrement() int32 {
	return atomic.AddInt32(&p.nextNodeId, 1)
}

// Appends given innovation to the list of known innovations in thread safe manner
func (p *Population) addInnovationSynced(i *Innovation) {
	p.mutex.Lock()
	p.Innovations = append(p.Innovations, i)
	p.mutex.Unlock()
}

// Create a population of size size off of Genome g. The new Population will have the same topology as g
// with link weights slightly perturbed from g's
func (p *Population) spawn(g *Genome, context *neat.NeatContext) (err error) {
	for count := 0; count < context.PopSize; count++ {
		// make genome duplicate for new organism
		new_genome, err := g.duplicate(count)
		if err != nil {
			return err
		}
		// introduce initial mutations
		if _, err = new_genome.mutateLinkWeights(1.0, 1.0, gaussianMutator); err != nil {
			return err
		}
		// create organism for new genome
		if new_organism, err := NewOrganism(0.0, new_genome, 1); err != nil {
			return err
		} else {
			p.Organisms = append(p.Organisms, new_organism)
		}
	}
	// Keep a record of the innovation and node number we are on
	if nextNodeId, err := g.getLastNodeId(); err != nil {
		return err
	} else {
		p.nextNodeId = int32(nextNodeId + 1)
	}
	if p.nextInnovNum, err = g.getNextGeneInnovNum(); err != nil {
		return err
	}

	// Separate the new Population into species
	err = p.speciate(p.Organisms, context)

	return err
}

// Check to see if the best species died somehow. We don't want this to happen!!!
// N.B. the mutated offspring of best species may be added to other more compatible species and as result
// the best species from previous generation will be removed, but their offspring still be alive.
// Returns error if best species died.
func (p *Population) checkBestSpeciesAlive(best_species_id int, best_species_reproduced bool) error {
	best_ok := false
	var best_sp_max_fitness float64
	for _, curr_species := range p.Species {
		neat.DebugLog(fmt.Sprintf("POPULATION: %d <> %d\n", curr_species.Id, best_species_id))
		if curr_species.Id == best_species_id {
			best_ok = true
			best_sp_max_fitness = curr_species.MaxFitnessEver
			break
		}
	}
	if !best_ok && !best_species_reproduced {
		return errors.New("POPULATION: The best species died without offspring!")
	} else {
		neat.DebugLog(fmt.Sprintf("POPULATION: The best survived species Id: %d, max fitness ever: %f",
			best_species_id, best_sp_max_fitness))
	}
	return nil
}

// Speciate separates given organisms into species of this population by checking compatibilities against a threshold.
// Any organism that does is not compatible with the first organism in any existing species becomes a new species.
func (p *Population) speciate(organisms []*Organism, context *neat.NeatContext) error {
	if len(organisms) == 0 {
		return errors.New("There is no organisms to speciate from")
	}

	// Step through all given organisms and speciate them within the population
	for _, curr_org := range organisms {
		if len(p.Species) == 0 {
			// Create the first species
			createFirstSpecies(p, curr_org)
		} else {
			if context.CompatThreshold == 0 {
				return errors.New("POPULATION: compatibility thershold is set to ZERO. " +
					"Will not find any compatible species.")
			}
			// For each organism, search for a species it is compatible to
			done := false
			var best_compatible *Species // the best compatible species
			best_compat_value := math.MaxFloat64
			for _, curr_species := range p.Species {
				comp_org := curr_species.firstOrganism()
				// compare current organism with first organism in current specie
				if comp_org != nil {
					curr_compat := curr_org.Genotype.compatibility(comp_org.Genotype, context)
					if curr_compat < context.CompatThreshold && curr_compat < best_compat_value {
						best_compatible = curr_species
						best_compat_value = curr_compat
						done = true
					}
				}
			}
			if done {
				neat.DebugLog(fmt.Sprintf("POPULATION: Compatible species [%d] found for baby organism [%d]",
					best_compatible.Id, curr_org.Genotype.Id))
				// Found compatible species, so add current organism to it
				best_compatible.addOrganism(curr_org);
				// Point organism to its species
				curr_org.Species = best_compatible
			} else {
				// If we didn't find a match, create a new species
				createFirstSpecies(p, curr_org)
			}
		}
	}

	return nil
}

// Removes zero offspring species from this population, i.e. species which will not have any offspring organism belonging to it
// after reproduction cycle due to its fitness stagnation
func (p *Population) purgeZeroOffspringSpecies(generation int) {
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
		sp.ExpectedOffspring, skim = sp.countOffspring(skim)
		total_expected += sp.ExpectedOffspring
	}
	neat.DebugLog(fmt.Sprintf("POPULATION: Total expected offspring count: %d", total_expected))

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
}

// When population stagnation detected the delta coding will be performed in attempt to fix this
func (p *Population) deltaCoding(sorted_species []*Species, context *neat.NeatContext) {
	neat.DebugLog("POPULATION: PERFORMING DELTA CODING TO FIX STAGNATION")
	p.EpochsHighestLastChanged = 0
	half_pop := context.PopSize / 2

	neat.DebugLog(fmt.Sprintf("half_pop: [%d] (pop_size - halfpop): [%d]\n",
		half_pop, context.PopSize - half_pop))

	curr_species := sorted_species[0]
	if len(sorted_species) > 1 {
		// Assign population to first two species
		curr_species.Organisms[0].superChampOffspring = half_pop
		curr_species.ExpectedOffspring = half_pop
		curr_species.AgeOfLastImprovement = curr_species.Age

		// process the second species
		curr_species = sorted_species[1]
		// NOTE: PopSize can be odd. That's why we use subtraction below
		curr_species.Organisms[0].superChampOffspring = context.PopSize - half_pop
		curr_species.ExpectedOffspring = context.PopSize - half_pop
		curr_species.AgeOfLastImprovement = curr_species.Age

		// Get rid of all species after the first two
		for i := 2; i < len(sorted_species); i++ {
			sorted_species[i].ExpectedOffspring = 0
		}
	} else {
		curr_species = sorted_species[0]
		curr_species.Organisms[0].superChampOffspring = context.PopSize
		curr_species.ExpectedOffspring = context.PopSize
		curr_species.AgeOfLastImprovement = curr_species.Age
	}
}

// The system can take expected offspring away from worse species and give them
// to superior species depending on the system parameter BabiesStolen (when BabiesStolen > 0)
func (p *Population) giveBabiesToTheBest(sorted_species []*Species, context *neat.NeatContext) {
	stolen_babies := 0 // Babies taken from the bad species and given to the champs

	curr_species := sorted_species[0] // the best species
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
			curr_species.Organisms[0].superChampOffspring = stolen_blocks[block_index]
			curr_species.ExpectedOffspring += stolen_blocks[block_index]
			stolen_babies -= stolen_blocks[block_index]
		} else if block_index >= 3 {
			// Give stolen to the rest in random ratios
			if rand.Float64() > 0.1 {
				// Randomize a little which species get boosted by a super champ
				if stolen_babies > 3 {
					curr_species.Organisms[0].superChampOffspring = 3
					curr_species.ExpectedOffspring += 3
					stolen_babies -= 3
				} else {
					curr_species.Organisms[0].superChampOffspring = stolen_babies
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
		curr_species.Organisms[0].superChampOffspring += stolen_babies
		curr_species.ExpectedOffspring += stolen_babies
	}
}

// Purge from population all organisms marked to be eliminated
func (p *Population) purgeOrganisms() error {
	org_to_keep := make([]*Organism, 0)
	for _, curr_org := range p.Organisms {
		if curr_org.toEliminate {
			// Remove the organism from its Species
			_, err := curr_org.Species.removeOrganism(curr_org)
			if err != nil {
				return err
			}
		} else {
			// Keep organism in population
			org_to_keep = append(org_to_keep, curr_org)
		}
	}
	// Keep only remained organisms in the population
	p.Organisms = org_to_keep

	return nil
}

// Destroy and remove the old generation of the organisms and of the species
func (p *Population) purgeOldGeneration(best_species_id int) error {
	for _, curr_org := range p.Organisms {
		// Remove the organism from its Species
		_, err := curr_org.Species.removeOrganism(curr_org)
		if err != nil {
			return err
		}

		if neat.LogLevel == neat.LogLevelDebug && curr_org.Species.Id == best_species_id {
			neat.DebugLog(fmt.Sprintf("POPULATION: Removed organism [%d] from best species [%d] - %d organisms remained",
				curr_org.Genotype.Id, best_species_id, len(curr_org.Species.Organisms)))
		}
	}
	p.Organisms = make([]*Organism, 0)

	return nil
}

// Removes all empty Species and age ones that survive.
// As this happens, create master organism list for the new generation.
func (p *Population) purgeOrAgeSpecies() {
	org_count := 0
	species_to_keep := make([]*Species, 0)
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
				curr_org.Genotype.Id = org_count
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
}
