package genetics

import (
	"github.com/yaricom/goNEAT/neat"
	"sort"
	"fmt"
	"errors"
	"encoding/gob"
	"bytes"
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
	sorted_species          []*Species
	best_species_reproduced bool
	best_species_id         int
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
	ex.sorted_species = nil

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
	ex.sorted_species = make([]*Species, len(p.Species))
	copy(ex.sorted_species, p.Species)

	// Sort the Species by max original fitness of its first organism
	sort.Sort(sort.Reverse(byOrganismOrigFitness(ex.sorted_species)))

	// Used in debugging to see why (if) best species dies
	ex.best_species_id = ex.sorted_species[0].Id

	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog("POPULATION: >> Sorted Species START <<")
		for _, sp := range ex.sorted_species {
			// Print out for Debugging/viewing what's going on
			neat.DebugLog(
				fmt.Sprintf("POPULATION: >> Orig. fitness of Species %d (Size %d): %f, current fitness: %f, expected offspring: %d, last improved %d \n",
					sp.Id, len(sp.Organisms), sp.Organisms[0].originalFitness, sp.Organisms[0].Fitness, sp.ExpectedOffspring, (sp.Age - sp.AgeOfLastImprovement)))
		}
		neat.DebugLog("POPULATION: >> Sorted Species END <<\n")

	}

	// Check for Population-level stagnation
	curr_species := ex.sorted_species[0]
	curr_species.Organisms[0].isPopulationChampion = true // DEBUG marker of the best of pop
	if curr_species.Organisms[0].originalFitness > p.HighestFitness {
		p.HighestFitness = curr_species.Organisms[0].originalFitness
		p.EpochsHighestLastChanged = 0
		neat.DebugLog(fmt.Sprintf("POPULATION: NEW POPULATION RECORD FITNESS: %f of SPECIES with ID: %d\n", p.HighestFitness, ex.best_species_id))

	} else {
		p.EpochsHighestLastChanged += 1
		neat.DebugLog(fmt.Sprintf(" generations since last population fitness record: %f\n", p.HighestFitness))
	}

	// Check for stagnation - if there is stagnation, perform delta-coding
	if p.EpochsHighestLastChanged >= context.DropOffAge + 5 {
		// Population stagnated - trying to fix it by delta coding
		p.deltaCoding(ex.sorted_species, context)
	} else if context.BabiesStolen > 0 {
		// STOLEN BABIES: The system can take expected offspring away from worse species and give them
		// to superior species depending on the system parameter BabiesStolen (when BabiesStolen > 0)
		p.giveBabiesToTheBest(ex.sorted_species, context)
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
		rep_babies, err := sp.reproduce(generation, p, ex.sorted_species, context)
		if err != nil {
			return err
		}
		if sp.Id == ex.best_species_id {
			// store flag if best species reproduced - it will be used to determine if best species
			// produced offspring before died
			ex.best_species_reproduced = true
		}

		// store babies
		babies = append(babies, rep_babies...)
	}

	// sanity check - make sure that population size keep the same
	if len(babies) != context.PopSize {
		return errors.New(
			fmt.Sprintf("POPULATION: Progeny size after reproduction cycle dimished.\nExpected: [%d], but got: [%d]",
				context.PopSize, len(babies)))
	}


	// speciate fresh progeny
	err := p.speciate(babies, context)

	neat.DebugLog("POPULATION: >>>>> Reproduction Complete")

	return err
}

func (ex *SequentialPopulationEpochExecutor) finalize(generation int, p *Population, context *neat.NeatContext) error {
	// Destroy and remove the old generation from the organisms and species
	err := p.purgeOldGeneration(ex.best_species_id)
	if err != nil {
		return err
	}

	// Removes all empty Species and age ones that survive.
	// As this happens, create master organism list for the new generation.
	p.purgeOrAgeSpecies()

	// Remove the innovations of the current generation
	p.Innovations = make([]*Innovation, 0)

	// Check to see if the best species died somehow. We don't want this to happen!!!
	err = p.checkBestSpeciesAlive(ex.best_species_id, ex.best_species_reproduced)

	// DEBUG: Checking the top organism's duplicate in the next gen
	// This prints the champ's child to the screen
	if neat.LogLevel == neat.LogLevelDebug && err != nil {
		for _, curr_org := range p.Organisms {
			if curr_org.isPopulationChampionChild {
				neat.DebugLog(
					fmt.Sprintf("POPULATION: At end of reproduction cycle, the child of the pop champ is: %s",
						curr_org.Genotype))
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
	sp_num := len(p.Species)
	res_chan := make(chan reproductionResult, sp_num)
	// The wait group to wait for all GO routines
	var wg sync.WaitGroup

	for _, curr_species := range p.Species {
		wg.Add(1)
		// run in separate GO thread
		go func(sp *Species, generation int, p *Population, sorted_species []*Species,
		context *neat.NeatContext, res_chan chan <- reproductionResult, wg *sync.WaitGroup) {

			babies, err := sp.reproduce(generation, p, sorted_species, context)
			res := reproductionResult{}
			if err == nil {
				res.species_id = sp.Id

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
					res.babies_stored = len(babies)
				}
			}
			res.err = err

			// write result to channel and signal to wait group
			res_chan <- res
			wg.Done()

		}(curr_species, generation, p, ex.sequential.sorted_species, context, res_chan, &wg)
	}

	// wait for reproduction results
	wg.Wait()
	close(res_chan)

	// read reproduction results, instantiate progeny and speciate over population
	babies := make([]*Organism, 0)
	for result := range res_chan {
		if result.err != nil {
			return result.err
		}
		// read baby genome
		dec := gob.NewDecoder(bytes.NewBuffer(result.babies))
		for i := 0; i < result.babies_stored; i++ {
			org := Organism{}
			err := dec.Decode(&org)
			if err != nil {
				return errors.New(
					fmt.Sprintf("POPULATION: Failed to decode baby organism, reason: %s", err))
			}
			babies = append(babies, &org)
		}
		if result.species_id == ex.sequential.best_species_id {
			// store flag if best species reproduced - it will be used to determine if best species
			// produced offspring before died
			ex.sequential.best_species_reproduced = (babies != nil)
		}
	}

	// sanity check - make sure that population size keep the same
	if len(babies) != context.PopSize {
		return errors.New(
			fmt.Sprintf("POPULATION: Progeny size after reproduction cycle dimished.\nExpected: [%d], but got: [%d]",
				context.PopSize, len(babies)))
	}


	// speciate fresh progeny
	err := p.speciate(babies, context)

	neat.DebugLog("POPULATION: >>>>> Reproduction Complete")

	return err
}