package genetics

import (
	"github.com/yaricom/goNEAT/neat"
	"sort"
	"math"
	"fmt"
	"errors"
	"math/rand"
	"io"
)

// A Species is a group of similar Organisms.
// Reproduction takes place mostly within a single species, so that compatible organisms can mate.
type Species struct {
	// The ID
	Id                   int;
	// The age of the Species
	Age                  int
	// The maximal fitness it ever had
	MaxFitnessEver       float64
	// How many child expected
	ExpectedOffspring    int

	// Is it novel
	IsNovel              bool

	// The organisms in the Species
	Organisms            Organisms
	// If this is too long ago, the Species will goes extinct
	AgeOfLastImprovement int

	// Flag used for search optimization
	IsChecked            bool
}

// Construct new species with specified ID
func NewSpecies(id int) *Species {
	return newSpecies(id)
}

// Allows the creation of a Species that won't age (a novel one). This protects new Species from aging
// inside their first generation
func NewSpeciesNovel(id int, novel bool) *Species {
	s := newSpecies(id)
	s.IsNovel = novel

	return s
}

// The private default constructor
func newSpecies(id int) *Species {
	return &Species{
		Id:id,
		Age:1,
		Organisms:make([]*Organism, 0),
	}
}

// Writes species to the specified writer
func (s Species) Write(w io.Writer) {
	_, avg := s.ComputeMaxAndAvgFitness()
	// Print a comment on the Species info
	fmt.Fprintf(w, "/* Species #%d : (Size %d) (AF %.3f) (Age %d)  */\n",
		s.Id, len(s.Organisms), avg, s.Age)

	// Sort organisms - best fitness first
	sorted_organisms := make(Organisms, len(s.Organisms))
	copy(sorted_organisms, s.Organisms)
	sort.Sort(sort.Reverse(sorted_organisms))

	// Print all the Organisms' Genomes to the outFile
	for _, org := range sorted_organisms {
		fmt.Fprintf(w, "/* Organism #%d Fitness: %.3f Error: %.3f */\n",
			org.Genotype.Id, org.Fitness, org.Error)
		if org.IsWinner {
			fmt.Fprintf(w, "/* ## $ WINNER ORGANISM FOR SPECIES #%d $ ## */\n", s.Id)
		}
		org.Genotype.Write(w)
	}
}


// Adds new Organism to the group related to this Species
func (s *Species) addOrganism(o *Organism) {
	s.Organisms = append(s.Organisms, o)
}
// Removes an organism from Species
func (s *Species) removeOrganism(org *Organism) (bool, error) {
	orgs := make([]*Organism, 0)
	for _, o := range s.Organisms {
		if o != org {
			orgs = append(orgs, o)
		}
	}
	if len(orgs) != len(s.Organisms) - 1 {
		return false, errors.New(
			fmt.Sprintf("SPECIES: Attempt to remove nonexistent Organism from Species with #of organisms: %d", len(s.Organisms)))
	} else {
		s.Organisms = orgs
		return true, nil
	}
}

// Can change the fitness of the organisms in the Species to be higher for very new species (to protect them).
// Divides the fitness by the size of the Species, so that fitness is "shared" by the species.
// NOTE: Invocation of this method will result of species organisms sorted by fitness in descending order, i.e. most fit will be first.
func (s *Species) adjustFitness(context *neat.NeatContext) {
	age_debt := (s.Age - s.AgeOfLastImprovement + 1) - context.DropOffAge
	if age_debt == 0 {
		age_debt = 1
	}

	for _, org := range s.Organisms {
		// Remember the original fitness before it gets modified
		org.originalFitness = org.Fitness

		// Make fitness decrease after a stagnation point dropoff_age
		// Added as if to keep species pristine until the dropoff point
		if age_debt >= 1 {
			// Extreme penalty for a long period of stagnation (divide fitness by 100)
			org.Fitness = org.Fitness * 0.01
		}

		// Give a fitness boost up to some young age (niching)
		// The age_significance parameter is a system parameter
		// if it is 1, then young species get no fitness boost
		if s.Age <= 10 {
			org.Fitness = org.Fitness * context.AgeSignificance
		}
		// Do not allow negative fitness
		if org.Fitness < 0.0 {
			org.Fitness = 0.0001
		}

		// Share fitness with the species
		org.Fitness = org.Fitness / float64(len(s.Organisms))
	}

	// Sort the population (most fit first) and mark for death those after : survival_thresh * pop_size
	sort.Sort(sort.Reverse(s.Organisms))

	// Update age_of_last_improvement here
	if s.Organisms[0].originalFitness > s.MaxFitnessEver {
		s.AgeOfLastImprovement = s.Age
		s.MaxFitnessEver = s.Organisms[0].originalFitness
	}

	// Decide how many get to reproduce based on survival_thresh * pop_size
	// Adding 1.0 ensures that at least one will survive
	num_parents := int(math.Floor(context.SurvivalThresh * float64(len(s.Organisms)) + 1.0))

	// Mark for death those who are ranked too low to be parents
	s.Organisms[0].isChampion = true // Mark the champ as such
	for c := num_parents; c < len(s.Organisms); c++ {
		s.Organisms[c].toEliminate = true
	}
}

// Computes maximal and average fitness of species
func (s Species) ComputeMaxAndAvgFitness() (max, avg float64) {
	total := 0.0
	for _, o := range s.Organisms {
		total += o.Fitness
		if o.Fitness > max {
			max = o.Fitness
		}
	}
	if len(s.Organisms) > 0 {
		avg = total / float64(len(s.Organisms))
	}
	return max, avg
}

// Returns most fit organism for this species
func (s Species) FindChampion() *Organism {
	champ_fitness := -1.0
	var champion *Organism
	for _, org := range s.Organisms {
		if org.Fitness > champ_fitness {
			champ_fitness = org.Fitness
			champion = org
		}
	}
	return champion
}

// Returns first organism or nil
func (s Species) firstOrganism() *Organism {
	if len(s.Organisms) > 0 {
		return s.Organisms[0]
	} else {
		return nil
	}
}

// Compute the collective offspring the entire species (the sum of all organism's offspring) is assigned.
// The skim is fractional offspring left over from a previous species that was counted. These fractional parts are
// kept until they add up to 1.
// Returns the whole offspring count for this species as well as fractional offspring left after computation (skim).
func (s Species) countOffspring(skim float64) (int, float64) {
	var e_o_intpart int  // The floor of an organism's expected offspring
	var e_o_fracpart float64 // Expected offspring fractional part
	var skim_intpart float64  // The whole offspring in the skim

	expectedOffspring := 0
	for _, o := range s.Organisms {
		e_o_intpart = int(math.Floor(o.ExpectedOffspring))
		e_o_fracpart = math.Mod(o.ExpectedOffspring, 1.0)

		expectedOffspring += e_o_intpart

		// Skim off the fractional offspring
		skim += e_o_fracpart

		if skim >= 1.0 {
			skim_intpart = math.Floor(skim)
			expectedOffspring += int(skim_intpart)
			skim -= skim_intpart
		}
	}
	return expectedOffspring, skim
}

// Compute generations since last improvement
func (s Species) lastImproved() int {
	return s.Age - s.AgeOfLastImprovement
}

// Returns size of this Species, i.e. number of Organisms belonging to it
func (s Species) size() int {
	return len(s.Organisms)
}

// Returns Organism - champion among others (best fitness)
func (s Species) findChampion() *Organism {
	sort.Sort(sort.Reverse(s.Organisms))
	return s.Organisms[0]
}

// Perform mating and mutation to form next generation. The sorted_species is ordered to have best species in the beginning.
// Returns list of baby organisms as a result of reproduction of all organisms in this species.
func (s Species) reproduce(generation int, pop *Population, sorted_species []*Species, context *neat.NeatContext) ([]*Organism, error) {
	//Check for a mistake
	if s.ExpectedOffspring > 0 && len(s.Organisms) == 0 {
		return nil, errors.New("SPECIES: ATTEMPT TO REPRODUCE OUT OF EMPTY SPECIES")
	}

	// The number of Organisms in the old generation
	pool_size := len(s.Organisms)
	// The champion of the 'this' specie is the first element of the specie;
	the_champ := s.Organisms[0]

	// The species babies
	babies := make([]*Organism, 0)

	// Flag the preservation of the champion
	champ_clone_done := false

	// Create the designated number of offspring for the Species one at a time
	for count := 0; count < s.ExpectedOffspring; count++ {
		neat.DebugLog(fmt.Sprintf("SPECIES: Offspring #%d from %d, (species: %d)",
			count, s.ExpectedOffspring, s.Id))

		mut_struct_baby, mate_baby := false, false

		// Debug Trap
		if s.ExpectedOffspring > context.PopSize {
			neat.WarnLog(fmt.Sprintf("SPECIES: Species [%d] expected offspring: %d exceeds population size limit: %d\n",
				s.Id, s.ExpectedOffspring, context.PopSize))
		}

		var baby *Organism

		if the_champ.superChampOffspring > 0 {
			neat.DebugLog("SPECIES: Reproduce super champion")

			// If we have a super_champ (Population champion), finish off some special clones
			mom := the_champ;
			new_genome, err := mom.Genotype.duplicate(count)
			if err != nil {
				return nil, err
			}

			// Most superchamp offspring will have their connection weights mutated only
			// The last offspring will be an exact duplicate of this super_champ
			// Note: Superchamp offspring only occur with stolen babies!
			//      Settings used for published experiments did not use this
			if the_champ.superChampOffspring > 1 {
				if rand.Float64() < 0.8 || context.MutateAddLinkProb == 0.0 {
					// Make sure no links get added when the system has link adding disabled
					new_genome.mutateLinkWeights(context.WeightMutPower, 1.0, gaussianMutator)
				} else {
					// Sometimes we add a link to a superchamp
					new_genome.Genesis(generation)
					if _, err = new_genome.mutateAddLink(pop, context); err != nil {
						return nil, err
					}
					mut_struct_baby = true;
				}
			}

			// Create the new baby organism
			baby, err = NewOrganism(0.0, new_genome, generation)
			if err != nil {
				return nil, err
			}

			if the_champ.superChampOffspring == 1 {
				if the_champ.isPopulationChampion {
					baby.isPopulationChampionChild = true
					baby.highestFitness = mom.originalFitness
				}
			}

			the_champ.superChampOffspring--
		} else if !champ_clone_done && s.ExpectedOffspring > 5 {
			neat.DebugLog("SPECIES: Clone species champion")

			// If we have a Species champion, just clone it
			mom := the_champ // Mom is the champ
			new_genome, err := mom.Genotype.duplicate(count)
			if err != nil {
				return nil, err
			}
			// Baby is just like mommy
			champ_clone_done = true

			// Create the new baby organism
			baby, err = NewOrganism(0.0, new_genome, generation)
			if err != nil {
				return nil, err
			}

		} else if rand.Float64() < context.MutateOnlyProb || pool_size == 1 {
			neat.DebugLog("SPECIES: Reproduce by applying random mutation:")

			// Apply mutations
			org_num := rand.Int31n(int32(pool_size)) // select random mom
			mom := s.Organisms[org_num]
			new_genome, err := mom.Genotype.duplicate(count)
			if err != nil {
				return nil, err
			}

			// Do the mutation depending on probabilities of various mutations
			if rand.Float64() < context.MutateAddNodeProb {
				neat.DebugLog("SPECIES: ---> mutateAddNode")

				// Mutate add node
				if _, err = new_genome.mutateAddNode(pop, context); err != nil {
					return nil, err
				}
				mut_struct_baby = true
			} else if rand.Float64() < context.MutateAddLinkProb {
				neat.DebugLog("SPECIES: ---> mutateAddLink")

				// Mutate add link
				new_genome.Genesis(generation)
				if _, err = new_genome.mutateAddLink(pop, context); err != nil {
					return nil, err
				}
				mut_struct_baby = true
			} else if rand.Float64() < context.MutateConnectSensors {
				neat.DebugLog("SPECIES: ---> mutateConnectSensors")
				if link_added, err := new_genome.mutateConnectSensors(pop, context); err != nil {
					return nil, err
				} else {
					mut_struct_baby = link_added
				}
			}

			if !mut_struct_baby {
				neat.DebugLog("SPECIES: ---> mutateAllNonstructural")

				// If we didn't do a structural mutation, we do the other kinds
				if _, err = new_genome.mutateAllNonstructural(context); err != nil {
					return nil, err
				}
			}

			// Create the new baby organism
			baby, err = NewOrganism(0.0, new_genome, generation);
			if err != nil {
				return nil, err
			}
		} else {
			neat.DebugLog("SPECIES: Reproduce by mating:")

			// Otherwise we should mate
			org_num := rand.Int31n(int32(pool_size)) // select random mom
			mom := s.Organisms[org_num]

			// Choose random dad
			var dad *Organism
			if rand.Float64() > context.InterspeciesMateRate {
				neat.DebugLog("SPECIES: ---> mate within species")

				// Mate within Species
				org_num = rand.Int31n(int32(pool_size))
				dad = s.Organisms[org_num]
			} else {
				neat.DebugLog("SPECIES: ---> mate outside species")

				// Mate outside Species
				rand_species := s

				// Select a random species
				giveup := 0
				for ; rand_species.Id == s.Id && giveup < 5; {
					// Choose a random species tending towards better species
					rand_mult := rand.Float64() / 4.0
					// This tends to select better species
					rand_species_num := int(math.Floor(rand_mult * float64(len(sorted_species))))
					rand_species = *sorted_species[rand_species_num]

					giveup++
				}
				dad = rand_species.Organisms[0]
			}

			// Perform mating based on probabilities of different mating types
			var new_genome *Genome
			var err error
			if rand.Float64() < context.MateMultipointProb {
				neat.DebugLog("SPECIES: ------> mateMultipoint")

				// mate multipoint baby
				new_genome, err = mom.Genotype.mateMultipoint(dad.Genotype, count, mom.originalFitness, dad.originalFitness)
				if err != nil {
					return nil, err
				}
			} else if rand.Float64() < context.MateMultipointAvgProb / (context.MateMultipointAvgProb + context.MateSinglepointProb) {
				neat.DebugLog("SPECIES: ------> mateMultipointAvg")

				// mate multipoint_avg baby
				new_genome, err = mom.Genotype.mateMultipointAvg(dad.Genotype, count, mom.originalFitness, dad.originalFitness)
				if err != nil {
					return nil, err
				}
			} else {
				neat.DebugLog("SPECIES: ------> mateSinglepoint")

				new_genome, err = mom.Genotype.mateSinglepoint(dad.Genotype, count)
				if err != nil {
					return nil, err
				}
			}

			mate_baby = true

			// Determine whether to mutate the baby's Genome
			// This is done randomly or if the mom and dad are the same organism
			if rand.Float64() > context.MateOnlyProb ||
				dad.Genotype.Id == mom.Genotype.Id ||
				dad.Genotype.compatibility(mom.Genotype, context) == 0.0 {
				neat.DebugLog("SPECIES: ------> Mutatte baby genome:")

				// Do the mutation depending on probabilities of  various mutations
				if rand.Float64() < context.MutateAddNodeProb {
					neat.DebugLog("SPECIES: ---------> mutateAddNode")

					// mutate_add_node
					if _, err = new_genome.mutateAddNode(pop, context); err != nil {
						return nil, err
					}
					mut_struct_baby = true
				} else if rand.Float64() < context.MutateAddLinkProb {
					neat.DebugLog("SPECIES: ---------> mutateAddLink")

					// mutate_add_link
					new_genome.Genesis(generation)
					if _, err = new_genome.mutateAddLink(pop, context); err != nil {
						return nil, err
					}
					mut_struct_baby = true
				} else if rand.Float64() < context.MutateConnectSensors {
					neat.DebugLog("SPECIES: ---> mutateConnectSensors")
					if link_added, err := new_genome.mutateConnectSensors(pop, context); err != nil {
						return nil, err
					} else {
						mut_struct_baby = link_added
					}
				}

				if !mut_struct_baby {
					neat.DebugLog("SPECIES: ---> mutateAllNonstructural")

					// If we didn't do a structural mutation, we do the other kinds
					if _, err := new_genome.mutateAllNonstructural(context); err != nil {
						return nil, err
					}
				}
			}
			// Create the new baby organism
			baby, err = NewOrganism(0.0, new_genome, generation)
			if err != nil {
				return nil, err
			}
		} // end else

		baby.mutationStructBaby = mut_struct_baby
		baby.mateBaby = mate_baby

		babies = append(babies, baby)

	} // end for count := 0
	return babies, nil
}

func createFirstSpecies(pop *Population, baby *Organism) {
	neat.DebugLog(fmt.Sprintf("SPECIES: Create first species for baby organism [%d]", baby.Genotype.Id))

	pop.LastSpecies++
	new_species := NewSpeciesNovel(pop.LastSpecies, true)
	pop.Species = append(pop.Species, new_species)
	new_species.addOrganism(baby) // Add the baby
	baby.Species = new_species // Point baby to its species

	neat.DebugLog(fmt.Sprintf("SPECIES: # of species in population: %d, new species id: %d",
		len(pop.Species), new_species.Id))
}

func (s Species) String() string {
	max, avg := s.ComputeMaxAndAvgFitness()
	str := fmt.Sprintf("Species #%d, age=%d, avg_fitness=%.3f, max_fitness=%.3f, max_fitness_ever=%.3f, expected_offspring=%d, age_of_last_improvement=%d\n",
		s.Id, s.Age, avg, max, s.MaxFitnessEver, s.ExpectedOffspring, s.AgeOfLastImprovement)
	str += fmt.Sprintf("Has %d Organisms:\n", len(s.Organisms))
	for _, o := range s.Organisms {
		str += fmt.Sprintf("\t%s\n", o)
	}
	return str
}

// This is used for list sorting of Species by original fitness of best organism highest fitness first
// It implements sort.Interface for []Species based on the OriginalFitness of first Organism field in descending order,
// i.e. the max fitness goes first
type byOrganismOrigFitness []*Species

func (f byOrganismOrigFitness) Len() int {
	return len(f)
}
func (f byOrganismOrigFitness) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}
func (f byOrganismOrigFitness) Less(i, j int) bool {
	org1 := f[i].Organisms[0]
	org2 := f[j].Organisms[0]
	if org1.originalFitness < org2.originalFitness {
		// try to promote most fit species
		return true // Lower fitness is less
	} else if org1.originalFitness == org2.originalFitness {
		// try to promote less complex species
		c1 := org1.Phenotype.Complexity()
		c2 := org2.Phenotype.Complexity()
		if c1 > c2 {
			return true // Higher complexity is "less"
		} else if c1 == c2 {
			// try to promote younger species
			return f[i].Age > f[j].Age // Higher Age is Less
		}
	}
	return false
}

// This is used for list sorting of species by maximal fitness
type ByOrganismFitness []*Species

func (f ByOrganismFitness) Len() int {
	return len(f)
}

func (f ByOrganismFitness) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}

func (f ByOrganismFitness) Less(i, j int) bool {
	i_max, _ := f[i].ComputeMaxAndAvgFitness()
	j_max, _ := f[j].ComputeMaxAndAvgFitness()
	return i_max < j_max
}