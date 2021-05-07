package genetics

import (
	"context"
	"errors"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
)

// A Population is a group of Organisms including their species
type Population struct {
	// Species in the Population. Note that the species should comprise all the genomes
	Species []*Species
	// The organisms in the Population
	Organisms []*Organism
	// The highest species number
	LastSpecies int
	// An integer that when above zero tells when the first winner appeared
	WinnerGen int
	// The last generation played
	FinalGen int

	// Stagnation detector
	HighestFitness float64
	// The number of epochs when highest fitness was recorded for this population. If it was too long before
	// than delta coding will be applied to avoid population's fitness stagnation
	EpochsHighestLastChanged int

	/* Fitness Statistics */
	MeanFitness float64
	Variance    float64
	StandardDev float64

	// For holding the genetic innovations of the newest generation
	innovations []Innovation
	// The next innovation number for population
	nextInnovNum int64
	// The next ID for new node in population
	nextNodeId int32

	// The mutex to guard against concurrent modifications
	mutex *sync.Mutex
}

// The auxiliary data type to hold results of parallel reproduction sent over the wires
type reproductionResult struct {
	babiesStored int    // the number of offsprings saved
	babies       []byte // the encoded offsprings
	err          error  // error occurred during reproduction if any
	speciesId    int    // the ID of species used for reproduction
}

// NewPopulation constructs off of a single spawning Genome
func NewPopulation(g *Genome, opts *neat.Options) (*Population, error) {
	if opts.PopSize <= 0 {
		return nil, fmt.Errorf("wrong population size in the context: %d", opts.PopSize)
	}

	pop := newPopulation()
	err := pop.spawn(g, opts)
	if err != nil {
		return nil, err
	}
	return pop, nil
}

// NewPopulationRandom is a special constructor to create a population of random topologies uses
// NewGenomeRand(new_id, in, out, n, maxHidden int, recurrent bool, link_prob float64)
// See the Genome constructor above for the argument specifications
func NewPopulationRandom(in, out, maxHidden int, recurrent bool, linkProb float64, opts *neat.Options) (*Population, error) {
	if opts.PopSize <= 0 {
		return nil, fmt.Errorf("wrong population size in the context: %d", opts.PopSize)
	}

	pop := newPopulation()
	for count := 0; count < opts.PopSize; count++ {
		gen := newGenomeRand(count, in, out, rand.Intn(maxHidden), maxHidden, recurrent, linkProb)
		org, err := NewOrganism(0.0, gen, 1)
		if err != nil {
			return nil, err
		}
		pop.Organisms = append(pop.Organisms, org)
	}
	pop.nextNodeId = int32(in + out + maxHidden + 1)
	pop.nextInnovNum = int64((in+out+maxHidden)*(in+out+maxHidden) + 1)

	err := pop.speciate(opts.NeatContext(), pop.Organisms)
	if err != nil {
		return nil, err
	}

	return pop, nil
}

// Verify is to run verification on all Genomes in this Population (Debugging)
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
		WinnerGen:                0,
		HighestFitness:           0.0,
		EpochsHighestLastChanged: 0,
		Species:                  make([]*Species, 0),
		Organisms:                make([]*Organism, 0),
		mutex:                    &sync.Mutex{},
	}
}

func (p *Population) NextNodeId() int {
	return int(atomic.AddInt32(&p.nextNodeId, 1))
}

func (p *Population) NextInnovationNumber() int64 {
	return atomic.AddInt64(&p.nextInnovNum, 1)
}

func (p *Population) StoreInnovation(innovation Innovation) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.innovations = append(p.innovations, innovation)
}

func (p *Population) Innovations() []Innovation {
	return p.innovations
}

// Create a population from Genome g. The new Population will have the same topology as g
// with link weights slightly perturbed from g's
func (p *Population) spawn(g *Genome, opts *neat.Options) (err error) {
	for count := 0; count < opts.PopSize; count++ {
		// make genome duplicate for new organism
		newGenome, err := g.duplicate(count)
		if err != nil {
			return err
		}
		// introduce initial mutations
		if _, err = newGenome.mutateLinkWeights(1.0, 1.0, gaussianMutator); err != nil {
			return err
		}
		// create organism for new genome
		if newOrganism, err := NewOrganism(0.0, newGenome, 1); err != nil {
			return err
		} else {
			p.Organisms = append(p.Organisms, newOrganism)
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
	} else {
		// to compensate +1 in gene next innovation
		p.nextInnovNum -= 1
	}

	// Separate the new Population into species
	err = p.speciate(opts.NeatContext(), p.Organisms)

	return err
}

// Check to see if the best species died somehow. We don't want this to happen!!!
// N.B. the mutated offspring of best species may be added to other more compatible species and as result
// the best species from previous generation will be removed, but their offspring still be alive.
// Returns error if best species died.
func (p *Population) checkBestSpeciesAlive(bestSpeciesId int, bestSpeciesReproduced bool) error {
	bestOk := false
	var bestSpMaxFitness float64
	for _, currSpecies := range p.Species {
		if neat.LogLevel == neat.LogLevelDebug {
			neat.DebugLog(fmt.Sprintf("POPULATION: %d <> %d\n", currSpecies.Id, bestSpeciesId))
		}

		if currSpecies.Id == bestSpeciesId {
			bestOk = true
			bestSpMaxFitness = currSpecies.MaxFitnessEver
			break
		}
	}
	if !bestOk && !bestSpeciesReproduced {
		return errors.New("best species died without offspring")
	} else if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf("POPULATION: The best survived species Id: %d, max fitness ever: %f",
			bestSpeciesId, bestSpMaxFitness))
	}
	return nil
}

// speciate separates given organisms into species of this population by checking compatibilities against a threshold.
// Any organism that does is not compatible with the first organism in any existing species becomes a new species.
func (p *Population) speciate(ctx context.Context, organisms []*Organism) error {
	if len(organisms) == 0 {
		return errors.New("no organisms to speciate from")
	}

	opts, found := neat.FromContext(ctx)
	if !found {
		return neat.ErrNEATOptionsNotFound
	}

	// Step through all given organisms and speciate them within the population
	for _, currOrg := range organisms {
		// check if context was canceled
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if len(p.Species) == 0 {
			// Create the first species
			createFirstSpecies(p, currOrg)
		} else {
			if opts.CompatThreshold == 0 {
				return errors.New("compatibility threshold is set to ZERO - will not find any compatible species")
			}
			// For each organism, search for a species it is compatible to
			done := false
			var bestCompatible *Species // the best compatible species
			bestCompatValue := math.MaxFloat64
			for _, currSpecies := range p.Species {
				compOrg := currSpecies.firstOrganism()
				// compare current organism with first organism in current specie
				if compOrg != nil {
					currCompat := currOrg.Genotype.compatibility(compOrg.Genotype, opts)
					if currCompat < opts.CompatThreshold && currCompat < bestCompatValue {
						bestCompatible = currSpecies
						bestCompatValue = currCompat
						done = true
					}
				}
			}
			if bestCompatible != nil && done {
				if neat.LogLevel == neat.LogLevelDebug {
					neat.DebugLog(fmt.Sprintf("POPULATION: Compatible species [%d] found for baby organism [%d]",
						bestCompatible.Id, currOrg.Genotype.Id))
				}
				// Found compatible species, so add current organism to it
				bestCompatible.addOrganism(currOrg)
				// Point organism to its species
				currOrg.Species = bestCompatible
			} else {
				// If we didn't find a match, create a new species
				createFirstSpecies(p, currOrg)
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
	totalOrganisms := len(p.Organisms)

	// Go through the organisms and add up their fitnesses to compute the overall average
	for _, o := range p.Organisms {
		total += o.Fitness
	}
	// The average modified fitness among ALL organisms
	overallAverage := total / float64(totalOrganisms)
	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf(
			"POPULATION: Generation %d: overall average fitness = %.3f, # of organisms: %d, # of species: %d\n",
			generation, overallAverage, len(p.Organisms), len(p.Species)))
	}

	// Now compute expected number of offspring for each individual organism
	if overallAverage != 0 {
		for _, o := range p.Organisms {
			o.ExpectedOffspring = o.Fitness / overallAverage
		}
	}

	//The fractional parts of expected offspring that can be used only when they accumulate above 1 for the purposes
	// of counting Offspring
	skim := 0.0
	// precision checking
	totalExpected := 0

	// Now add those offspring up within each Species to get the number of offspring per Species
	for _, sp := range p.Species {
		sp.ExpectedOffspring, skim = sp.countOffspring(skim)
		totalExpected += sp.ExpectedOffspring
	}
	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf("POPULATION: Total expected offspring count: %d", totalExpected))
	}

	// Need to make up for lost floating point precision in offspring assignment.
	// If we lost precision, give an extra baby to the best Species
	if totalExpected < totalOrganisms {
		// Find the Species expecting the most
		var bestSpecies *Species
		maxExpected := 0
		finalExpected := 0
		for _, sp := range p.Species {
			if sp.ExpectedOffspring >= maxExpected {
				maxExpected = sp.ExpectedOffspring
				bestSpecies = sp
			}
			finalExpected += sp.ExpectedOffspring
		}
		// Give the extra offspring to the best species
		if bestSpecies != nil {
			bestSpecies.ExpectedOffspring += 1
		}
		finalExpected++

		// If we still aren't at total, there is a problem. Note that this can happen if a stagnant Species
		// dominates the population and then gets killed off by its age. Then the whole population plummets in
		// fitness. If the average fitness is allowed to hit 0, then we no longer have an average we can use to
		// assign offspring.
		if finalExpected < totalOrganisms {
			if neat.LogLevel == neat.LogLevelDebug {
				neat.DebugLog(fmt.Sprintf("POPULATION: Population died !!! (expected/total) %d/%d", finalExpected, totalOrganisms))
			}
			for _, sp := range p.Species {
				sp.ExpectedOffspring = 0
			}
			if bestSpecies != nil {
				bestSpecies.ExpectedOffspring = totalOrganisms
			}
		}
	}

	// Remove stagnated species which can not produce any offspring any more
	speciesToKeep := make([]*Species, 0)
	for _, sp := range p.Species {
		if sp.ExpectedOffspring > 0 {
			speciesToKeep = append(speciesToKeep, sp)
		}
	}
	p.Species = speciesToKeep
}

// When population stagnation detected the delta coding will be performed in attempt to fix this
func (p *Population) deltaCoding(sortedSpecies []*Species, opts *neat.Options) {
	neat.DebugLog("POPULATION: PERFORMING DELTA CODING TO FIX STAGNATION")
	p.EpochsHighestLastChanged = 0
	halfPop := opts.PopSize / 2

	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf("half_pop: [%d] (pop_size - halfpop): [%d]\n",
			halfPop, opts.PopSize-halfPop))
	}

	currSpecies := sortedSpecies[0]
	if len(sortedSpecies) > 1 {
		// Assign population to first two species
		currSpecies.Organisms[0].superChampOffspring = halfPop
		currSpecies.ExpectedOffspring = halfPop
		currSpecies.AgeOfLastImprovement = currSpecies.Age

		// process the second species
		currSpecies = sortedSpecies[1]
		// NOTE: PopSize can be odd. That's why we use subtraction below
		currSpecies.Organisms[0].superChampOffspring = opts.PopSize - halfPop
		currSpecies.ExpectedOffspring = opts.PopSize - halfPop
		currSpecies.AgeOfLastImprovement = currSpecies.Age

		// Get rid of all species after the first two
		for i := 2; i < len(sortedSpecies); i++ {
			sortedSpecies[i].ExpectedOffspring = 0
		}
	} else {
		currSpecies = sortedSpecies[0]
		currSpecies.Organisms[0].superChampOffspring = opts.PopSize
		currSpecies.ExpectedOffspring = opts.PopSize
		currSpecies.AgeOfLastImprovement = currSpecies.Age
	}
}

// The system can take expected offspring away from worse species and give them
// to superior species depending on the system parameter BabiesStolen (when BabiesStolen > 0)
func (p *Population) giveBabiesToTheBest(sortedSpecies []*Species, opts *neat.Options) {
	stolenBabies := 0 // Babies taken from the bad species and given to the champs

	// Take away a constant number of expected offspring from the worst few species
	for i := len(sortedSpecies) - 1; i >= 0 && stolenBabies < opts.BabiesStolen; i-- {
		currSpecies := sortedSpecies[i]
		if currSpecies.Age > 5 && currSpecies.ExpectedOffspring > 2 {
			if currSpecies.ExpectedOffspring-1 >= opts.BabiesStolen-stolenBabies {
				// This species has enough to finish off the stolen pool
				currSpecies.ExpectedOffspring -= opts.BabiesStolen - stolenBabies
				stolenBabies = opts.BabiesStolen
			} else {
				// Not enough here to complete the pool of stolen
				stolenBabies += currSpecies.ExpectedOffspring - 1
				currSpecies.ExpectedOffspring = 1
			}
		}
	}

	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf("POPULATION: STOLEN BABIES: %d\n", stolenBabies))
	}

	// Mark the best champions of the top species to be the super champs who will take on the extra
	// offspring for cloning or mutant cloning.
	// Determine the exact number that will be given to the top three.
	// They will get, in order, 1/5 1/5 and 1/10 of the stolen babies
	stolenBlocks := []int{opts.BabiesStolen / 5, opts.BabiesStolen / 5, opts.BabiesStolen / 10}
	blockIndex := 0
	for _, currSpecies := range sortedSpecies {
		if currSpecies.lastImproved() > opts.DropOffAge {
			// Don't give a chance to dying species even if they are champs
			continue
		}

		if blockIndex < 3 && stolenBabies >= stolenBlocks[blockIndex] {
			// Give stolen babies to the top three in 1/5 1/5 and 1/10 ratios
			currSpecies.Organisms[0].superChampOffspring = stolenBlocks[blockIndex]
			currSpecies.ExpectedOffspring += stolenBlocks[blockIndex]
			stolenBabies -= stolenBlocks[blockIndex]
		} else if blockIndex >= 3 {
			// Give stolen to the rest in random ratios
			if rand.Float64() > 0.1 {
				// Randomize a little which species get boosted by a super champ
				if stolenBabies > 3 {
					currSpecies.Organisms[0].superChampOffspring = 3
					currSpecies.ExpectedOffspring += 3
					stolenBabies -= 3
				} else {
					currSpecies.Organisms[0].superChampOffspring = stolenBabies
					currSpecies.ExpectedOffspring += stolenBabies
					stolenBabies = 0
				}
			}
		}

		if stolenBabies <= 0 {
			break
		}
		blockIndex++
	}
	// If any stolen babies aren't taken, give them to species #1's champ
	if stolenBabies > 0 {
		currSpecies := sortedSpecies[0]
		currSpecies.Organisms[0].superChampOffspring += stolenBabies
		currSpecies.ExpectedOffspring += stolenBabies
	}
}

// Purge from population all organisms marked to be eliminated
func (p *Population) purgeOrganisms() error {
	orgToKeep := make([]*Organism, 0)
	for _, org := range p.Organisms {
		if org.toEliminate {
			// Remove the organism from its Species
			_, err := org.Species.removeOrganism(org)
			if err != nil {
				return err
			}
		} else {
			// Keep organism in population
			orgToKeep = append(orgToKeep, org)
		}
	}
	// Keep only remained organisms in the population
	p.Organisms = orgToKeep

	return nil
}

// Destroy and remove the old generation of the organisms and of the species
func (p *Population) purgeOldGeneration(bestSpeciesId int) error {
	for _, org := range p.Organisms {
		// Remove the organism from its Species
		_, err := org.Species.removeOrganism(org)
		if err != nil {
			return err
		}

		if neat.LogLevel == neat.LogLevelDebug && org.Species.Id == bestSpeciesId {
			neat.DebugLog(fmt.Sprintf("POPULATION: Removed organism [%d] from best species [%d] - %d organisms remained",
				org.Genotype.Id, bestSpeciesId, len(org.Species.Organisms)))
		}
	}
	p.Organisms = make([]*Organism, 0)

	return nil
}

// Removes all empty Species and age ones that survive.
// As this happens, create master organism list for the new generation.
func (p *Population) purgeOrAgeSpecies() {
	orgCount := 0
	speciesToKeep := make([]*Species, 0)
	for _, currSpecies := range p.Species {
		if len(currSpecies.Organisms) > 0 {
			// Age surviving Species
			if currSpecies.IsNovel {
				currSpecies.IsNovel = false
			} else {
				currSpecies.Age += 1
			}
			// Rebuild master Organism list of population: NUMBER THEM as they are added to the list
			for _, org := range currSpecies.Organisms {
				org.Genotype.Id = orgCount
				p.Organisms = append(p.Organisms, org)
				orgCount++
			}
			// keep this species
			speciesToKeep = append(speciesToKeep, currSpecies)
		} else if neat.LogLevel == neat.LogLevelDebug {
			neat.DebugLog(fmt.Sprintf("POPULATION: >> Species [%d] have not survived reproduction!",
				currSpecies.Id))
		}
	}
	// Keep only survived species
	p.Species = speciesToKeep

	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf("POPULATION: # of species survived: %d, # of organisms survived: %d\n",
			len(p.Species), len(p.Organisms)))
	}
}
