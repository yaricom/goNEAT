package genetics

import (
	"github.com/yaricom/goNEAT/neat/network"
	"fmt"
)

// Organisms are Genomes and Networks with fitness information,
// i.e. the genotype and phenotype together.
type Organism struct {
	// A measure of fitness for the Organism
	Fitness              float64
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
	return (*f[i]).Fitness < (*f[j]).Fitness // lower fitness is less
}
