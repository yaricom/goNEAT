package genetics

import (
	"github.com/yaricom/goNEAT/neat/network"
	"fmt"
)

// Organisms are Genomes and Networks with fitness information,
// i.e. the genotype and phenotype together.
type Organism struct {
	// A measure of fitness for the Organism
	Fitness                   float64
	// A fitness measure that won't change during adjustments
	OriginalFitness           float64

	// Win marker (if needed for a particular task)
	IsWinner                  bool

	// The Organism's phenotype
	Net                       *network.Network
	// The Organism's genotype
	GNome                     *Genome
	// The Species of the Organism
	SpeciesOf                 *Species

	// Number of children this Organism may have
	ExpectedOffspring         float64
	// Tells which generation this Organism is from
	Generation                int

	// Marker for destruction of inferior Organisms
	ToEliminate               bool
	// Marks the species champion
	IsChampion                bool

	// Number of reserved offspring for a population leader
	SuperChampOffspring       int
	// Marks the best in population
	IsPopulationChampion      bool
	// Marks the duplicate child of a champion (for tracking purposes)
	IsPopulationChampionChild bool

	// DEBUG variable - highest fitness of champ
	highestFitness            float64

	// Track its origin - for debugging or analysis - we can tell how the organism was born
	mutationStructBaby        bool
	mateBaby                  bool

	// Used just for reporting purposes
	error float64
}

// Creates new organism with specified genome, fitness and given generation number
func NewOrganism(fit float64, g *Genome, generation int) *Organism {
	return &Organism{
		Fitness:fit,
		GNome:g,
		Net:g.Phenotype, // TODO g.genessis(g.Id)
		Generation:generation,
	}
}

// Regenerate the network based on a change in the genotype
func (o *Organism) UpdatePhenotype() {
	// First, delete the old phenotype (net)
	o.Net = nil

	// Now, recreate the phenotype off the new genotype
	o.Net = o.GNome.genesis(o.GNome.Id)
}

func (o *Organism) String() string {
	champStr := ""
	if o.IsChampion {
		champStr = " - CHAMPION - "
	}
	eliminStr := ""
	if o.ToEliminate {
		eliminStr = " - TO BE ELIMINATED - "
	}
	return fmt.Sprintf("[Organism generation: %d, fitness: %.3f, original fitness: %.3f%s%s]",
		o.Generation, o.Fitness, o.OriginalFitness, champStr, eliminStr)
}

// ByFitness implements sort.Interface for []Organism based on the Fitness field in descending order.
type ByFitness []*Organism
func (f ByFitness) Len() int {
	return len(f)
}
func (f ByFitness) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}
func (f ByFitness) Less(i, j int) bool {
	return f[i].Fitness > f[j].Fitness
}
