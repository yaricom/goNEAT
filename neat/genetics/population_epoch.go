package genetics

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"sort"
	"sync"
)

// The epoch executor type definition
type EpochExecutorType int

const (
	// The sequential executor
	SequentialExecutorType EpochExecutorType = 0
	// The parallel executor to perform reproduction cycle in parallel threads
	ParallelExecutorType = 1
)

// Executes epoch's turnover for population of organisms
type PopulationEpochExecutor interface {
	// Turnover the population to a new generation
	NextEpoch(generation int, population *Population, context *neat.NeatContext) error
}

// The epoch executor which will run execution sequentially in single thread for all species and organisms
type SequentialPopulationEpochExecutor struct {
	sortedSpecies         []*Species
	bestSpeciesReproduced bool
	bestSpeciesId         int
}

func (ex *SequentialPopulationEpochExecutor) NextEpoch(generation int, population *Population, context *neat.NeatContext) error {
	err := ex.prepare(generation, population, context)
	if err != nil {
		return err
	}
	err = ex.reproduce(generation, population, context)
	if err != nil {
		return err
	}
	err = ex.finalize(generation, population, context)

	neat.DebugLog(fmt.Sprintf("POPULATION: >>>>> Epoch %d complete\n", generation))

	return err
}

func (ex *SequentialPopulationEpochExecutor) prepare(generation int, p *Population, context *neat.NeatContext) error {
	// clear executor state from previous run
	ex.sortedSpecies = nil

	// Use Species' ages to modify the objective fitness of organisms in other words, make it more fair for younger
	// species so they have a chance to take hold and also penalize stagnant species. Then adjust the fitness using
	// the species size to "share" fitness within a species. Then, within each Species, mark for death those below
	// survival_thresh * average
	for _, sp := range p.Species {
		sp.adjustFitness(context)
	}

	// find and remove species unable to produce offspring due to fitness stagnation
	p.purgeZeroOffspringSpecies(generation)

	// Stick the Species pointers into a new Species list for sorting
	ex.sortedSpecies = make([]*Species, len(p.Species))
	copy(ex.sortedSpecies, p.Species)

	// Sort the Species by max original fitness of its first organism
	sort.Sort(sort.Reverse(byOrganismOrigFitness(ex.sortedSpecies)))

	// Used in debugging to see why (if) best species dies
	ex.bestSpeciesId = ex.sortedSpecies[0].Id

	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog("POPULATION: >> Sorted Species START <<")
		for _, sp := range ex.sortedSpecies {
			// Print out for Debugging/viewing what's going on
			neat.DebugLog(
				fmt.Sprintf("POPULATION: >> Orig. fitness of Species %d (Size %d): %f, current fitness: %f, expected offspring: %d, last improved %d \n",
					sp.Id, len(sp.Organisms), sp.Organisms[0].originalFitness, sp.Organisms[0].Fitness, sp.ExpectedOffspring,
					sp.Age-sp.AgeOfLastImprovement))
		}
		neat.DebugLog("POPULATION: >> Sorted Species END <<\n")
	}

	// Check for Population-level stagnation
	currSpecies := ex.sortedSpecies[0]
	currSpecies.Organisms[0].isPopulationChampion = true // DEBUG marker of the best of pop
	if currSpecies.Organisms[0].originalFitness > p.HighestFitness {
		p.HighestFitness = currSpecies.Organisms[0].originalFitness
		p.EpochsHighestLastChanged = 0
		if neat.LogLevel == neat.LogLevelDebug {
			neat.DebugLog(fmt.Sprintf("POPULATION: NEW POPULATION RECORD FITNESS: %f of SPECIES with ID: %d\n", p.HighestFitness, ex.bestSpeciesId))
		}
	} else {
		p.EpochsHighestLastChanged += 1
		if neat.LogLevel == neat.LogLevelDebug {
			neat.DebugLog(fmt.Sprintf(" generations since last population fitness record: %f\n", p.HighestFitness))
		}
	}

	// Check for stagnation - if there is stagnation, perform delta-coding
	if p.EpochsHighestLastChanged >= context.DropOffAge+5 {
		// Population stagnated - trying to fix it by delta coding
		p.deltaCoding(ex.sortedSpecies, context)
	} else if context.BabiesStolen > 0 {
		// STOLEN BABIES: The system can take expected offspring away from worse species and give them
		// to superior species depending on the system parameter BabiesStolen (when BabiesStolen > 0)
		p.giveBabiesToTheBest(ex.sortedSpecies, context)
	}

	// Kill off all Organisms marked for death. The remainder will be allowed to reproduce.
	err := p.purgeOrganisms()
	return err
}

// Do sequential reproduction cycle
func (ex *SequentialPopulationEpochExecutor) reproduce(generation int, p *Population, context *neat.NeatContext) error {
	neat.DebugLog("POPULATION: Start Sequential Reproduction Cycle >>>>>")

	// Perform reproduction. Reproduction is done on a per-Species basis
	babies := make([]*Organism, 0)

	for _, sp := range p.Species {
		repBabies, err := sp.reproduce(generation, p, ex.sortedSpecies, context)
		if err != nil {
			return err
		}
		if sp.Id == ex.bestSpeciesId {
			// store flag if best species reproduced - it will be used to determine if best species
			// produced offspring before died
			ex.bestSpeciesReproduced = true
		}

		// store babies
		babies = append(babies, repBabies...)
	}

	// sanity check - make sure that population size keep the same
	if len(babies) != context.PopSize {
		return fmt.Errorf("progeny size after reproduction cycle dimished.\nExpected: [%d], but got: [%d]",
			context.PopSize, len(babies))
	}

	// speciate fresh progeny
	err := p.speciate(babies, context)

	neat.DebugLog("POPULATION: >>>>> Reproduction Complete")

	return err
}

func (ex *SequentialPopulationEpochExecutor) finalize(generation int, p *Population, _ *neat.NeatContext) error {
	// Destroy and remove the old generation from the organisms and species
	err := p.purgeOldGeneration(ex.bestSpeciesId)
	if err != nil {
		return err
	}

	// Removes all empty Species and age ones that survive.
	// As this happens, create master organism list for the new generation.
	p.purgeOrAgeSpecies()

	// Remove the innovations of the current generation
	p.Innovations = make([]*Innovation, 0)

	// Check to see if the best species died somehow. We don't want this to happen!!!
	err = p.checkBestSpeciesAlive(ex.bestSpeciesId, ex.bestSpeciesReproduced)

	// DEBUG: Checking the top organism's duplicate in the next gen
	// This prints the champ's child to the screen
	if neat.LogLevel == neat.LogLevelDebug && err != nil {
		for _, org := range p.Organisms {
			if org.isPopulationChampionChild {
				neat.DebugLog(fmt.Sprintf("POPULATION: At end of reproduction cycle, the child of the pop champ is: %s",
					org.Genotype))
			}
		}
	}
	return err
}

// The population epoch executor with parallel reproduction cycle
type ParallelPopulationEpochExecutor struct {
	sequential *SequentialPopulationEpochExecutor
}

func (ex *ParallelPopulationEpochExecutor) NextEpoch(generation int, population *Population, context *neat.NeatContext) error {
	ex.sequential = &SequentialPopulationEpochExecutor{}
	err := ex.sequential.prepare(generation, population, context)
	if err != nil {
		return err
	}

	// Do parallel reproduction
	err = ex.reproduce(generation, population, context)
	if err != nil {
		return err
	}

	err = ex.sequential.finalize(generation, population, context)

	neat.DebugLog(fmt.Sprintf("POPULATION: >>>>> Epoch %d complete\n", generation))

	return err
}

// Do parallel reproduction cycle
func (ex *ParallelPopulationEpochExecutor) reproduce(generation int, p *Population, context *neat.NeatContext) error {
	neat.DebugLog("POPULATION: Start Parallel Reproduction Cycle >>>>>")

	// Perform reproduction. Reproduction is done on a per-Species basis
	spNum := len(p.Species)
	resChan := make(chan reproductionResult, spNum)
	// The wait group to wait for all GO routines
	var wg sync.WaitGroup

	for _, species := range p.Species {
		wg.Add(1)
		// run in separate GO thread
		go func(sp *Species, generation int, p *Population, sortedSpecies []*Species,
			context *neat.NeatContext, resChan chan<- reproductionResult, wg *sync.WaitGroup) {

			babies, err := sp.reproduce(generation, p, sortedSpecies, context)
			res := reproductionResult{}
			if err == nil {
				res.speciesId = sp.Id

				// fill babies into result
				var buf bytes.Buffer
				enc := gob.NewEncoder(&buf)
				for _, baby := range babies {
					err = enc.Encode(baby)
					if err != nil {
						break
					}
				}
				if err == nil {
					res.babies = buf.Bytes()
					res.babiesStored = len(babies)
				}
			}
			res.err = err

			// write result to channel and signal to wait group
			resChan <- res
			wg.Done()

		}(species, generation, p, ex.sequential.sortedSpecies, context, resChan, &wg)
	}

	// wait for reproduction results
	wg.Wait()
	close(resChan)

	// read reproduction results, instantiate progeny and speciate over population
	babies := make([]*Organism, 0)
	for result := range resChan {
		if result.err != nil {
			return result.err
		}
		// read baby genome
		dec := gob.NewDecoder(bytes.NewBuffer(result.babies))
		for i := 0; i < result.babiesStored; i++ {
			org := Organism{}
			err := dec.Decode(&org)
			if err != nil {
				return fmt.Errorf("failed to decode baby organism, reason: %v", err)
			}
			babies = append(babies, &org)
		}
		if result.speciesId == ex.sequential.bestSpeciesId {
			// store flag if best species reproduced - it will be used to determine if best species
			// produced offspring before died
			ex.sequential.bestSpeciesReproduced = babies != nil
		}
	}

	// sanity check - make sure that population size keep the same
	if len(babies) != context.PopSize {
		return fmt.Errorf("rogeny size after reproduction cycle dimished.\nExpected: [%d], but got: [%d]",
			context.PopSize, len(babies))
	}

	// speciate fresh progeny
	err := p.speciate(babies, context)

	neat.DebugLog("POPULATION: >>>>> Reproduction Complete")

	return err
}
