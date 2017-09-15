package genetics

import (
	"github.com/yaricom/goNEAT/neat"
	"sort"
	"math"
	"fmt"
)

// A Species is a group of similar Organisms.
// Reproduction takes place mostly within a single species, so that compatible organisms can mate.
type Species struct {
	// The ID
	Id                   int;
	// The age of the Species
	Age                  int
	// The average fitness of the Species
	AvgFitness           float64
	// The maximal fitness of the Species
	MaxFitness           float64
	// The maximal fitness it ever had
	MaxFitnessEver       float64
	// How many child expected
	ExpectedOffspring    int

	// Is it novel
	IsNovel              bool
	// Is it tested
	IsChecked            bool

	// The organisms in the Species
	Organisms            []*Organism
	// If this is too long ago, the Species will goes extinct
	AgeOfLastImprovement int
}

// Construct new species with specified ID
func NewSpecies(id int) *Species  {
	return newSpecies(id)
}

// Allows the creation of a Species that won't age (a novel one). This protects new Species from aging
// inside their first generation
func NewSpeciesNovel(id int, novel bool) *Species  {
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


// Adds new Organism to the group related to this Species
func (s *Species) AddOrganism(o *Organism) {
	s.Organisms = append(s.Organisms, o)
}
 // Returns first Organism from this Species
func (s *Species) FirstOrganism() *Organism {
	if len(s.Organisms) > 0 {
		return s.Organisms[0]
	} else {
		return nil
	}
}

// Can change the fitness of the organisms in the Species to be higher for very new species (to protect them).
// Divides the fitness by the size of the Species, so that fitness is "shared" by the species.
func (s *Species) AdjustFitness(conf *neat.Neat) {
	age_debt := (s.Age - s.AgeOfLastImprovement + 1) - conf.DropOffAge
	if age_debt == 0 {
		age_debt =1
	}

	for _, org := range s.Organisms {
		// Remember the original fitness before it gets modified
		org.OriginalFitness = org.Fitness

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
			org.Fitness = org.Fitness * conf.AgeSignificance
		}
		//Do not allow negative fitness
		if org.Fitness < 0.0 {
			org.Fitness = 0.0001
		}
		// Share fitness with the species
		org.Fitness = org.Fitness / float64(len(s.Organisms))
	}

	// Sort the population (most fit first) and mark for death those after : survival_thresh * pop_size
	sort.Reverse(ByFitness(s.Organisms))

	// Update age_of_last_improvement here
	if s.Organisms[0].OriginalFitness > s.MaxFitnessEver {
		s.AgeOfLastImprovement = s.Age
		s.MaxFitnessEver = s.Organisms[0].OriginalFitness
	}

	// Decide how many get to reproduce based on survival_thresh * pop_size
	// Adding 1.0 ensures that at least one will survive
	num_parents := int(math.Floor(conf.SurvivalThresh * float64(len(s.Organisms)) + 1.0))

	//Mark for death those who are ranked too low to be parents
	s.Organisms[0].IsChampion = true // Mark the champ as such
	for c := num_parents; c < len(s.Organisms); c++ {
		s.Organisms[c].ToEliminate = true
	}
}