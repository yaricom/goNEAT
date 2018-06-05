package genetics

import (
	"github.com/yaricom/goNEAT/neat/network"
	"fmt"
)

// The object to associate implementation specific data with particular organism for various algorithm implementations
type OrganismData struct {
	// The implementation specific data object to be associated with organism
	Value interface{}
}

// Organisms are Genotypes (Genomes) and Phenotypes (Networks) with fitness information,
// i.e. the genotype and phenotype together.
type Organism struct {
	// A measure of fitness for the Organism
	Fitness                   float64
	// A fitness measure that won't change during adjustments
	OriginalFitness           float64

	// Win marker (if needed for a particular task)
	IsWinner                  bool

	// The Organism's phenotype
	Phenotype                 *network.Network
	// The Organism's genotype
	Genotype                  *Genome
	// The Species of the Organism
	Species                   *Species

	// Number of children this Organism may have
	ExpectedOffspring         float64
	// Tells which generation this Organism is from
	Generation                int

	// The utility data transfer object to be used by different GA implementations to hold additional data.
	// Implemented as ANY to allow implementation specific objects.
	Data                      *OrganismData

	// Marker for destruction of inferior Organisms
	toEliminate               bool
	// Marks the species champion
	isChampion                bool

	// Number of reserved offspring for a population leader
	superChampOffspring       int
	// Marks the best in population
	isPopulationChampion      bool
	// Marks the duplicate child of a champion (for tracking purposes)
	isPopulationChampionChild bool

	// DEBUG variable - highest fitness of champ
	highestFitness            float64

	// Track its origin - for debugging or analysis - we can tell how the organism was born
	mutationStructBaby        bool
	mateBaby                  bool

	// Used just for reporting purposes
	Error                     float64
}

// Creates new organism with specified genome, fitness and given generation number
func NewOrganism(fit float64, g *Genome, generation int) *Organism {
	return &Organism{
		Fitness:fit,
		Genotype:g,
		Phenotype:g.genesis(g.Id),
		Generation:generation,
	}
}

// Regenerate the network based on a change in the genotype
func (o *Organism) UpdatePhenotype() {
	// First, delete the old phenotype (net)
	o.Phenotype = nil

	// Now, recreate the phenotype off the new genotype
	o.Phenotype = o.Genotype.genesis(o.Genotype.Id)
}

// Method to check if this algorithm is champion child and if so than if it's damaged
func (o *Organism) CheckChampionChildDamaged() bool {
	if o.isPopulationChampionChild && o.highestFitness > o.Fitness {
		return true
	}
	return false
}

func (o *Organism) String() string {
	champStr := ""
	if o.isChampion {
		champStr = " - CHAMPION - "
	}
	eliminStr := ""
	if o.toEliminate {
		eliminStr = " - TO BE ELIMINATED - "
	}
	return fmt.Sprintf("[Organism generation: %d, fitness: %.3f, original fitness: %.3f%s%s]",
		o.Generation, o.Fitness, o.OriginalFitness, champStr, eliminStr)
}

// Organisms is sortable list of organisms by fitness
type Organisms []*Organism

func (f Organisms) Len() int {
	return len(f)
}
func (f Organisms) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}
func (f Organisms) Less(i, j int) bool {
	if f[i].Fitness < f[j].Fitness {
		// try to promote most fit organisms
		return true  // lower fitness is less
	} else if f[i].Fitness == f[j].Fitness {
		// try to promote less complex organisms
		ci := f[i].Phenotype.Complexity()
		cj := f[j].Phenotype.Complexity()
		if ci > cj {
			return true // higher complexity is less
		} else if ci == cj {
			return f[i].Genotype.Id < f[j].Genotype.Id // least recent (older) is less
		}
	}
	return false
}
