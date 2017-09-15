package genetics

import "github.com/yaricom/goNEAT/neat/network"

// Organisms are Genomes and Networks with fitness information,
// i.e. the genotype and phenotype together.
type Organism struct {
	// A measure of fitness for the Organism
	Fitness float64
	// A fitness measure that won't change during adjustments
	OriginalFitness float64

	// Win marker (if needed for a particular task)
	IsWinner bool

	// The Organism's phenotype
	Net *network.Network
	// The Organism's genotype
	GNome *Genome
	// The Species of the Organism
	SpeciesOf *Species

	// Number of children this Organism may have
	ExpectedOffspring int64
	// Tells which generation this Organism is from
	Generation int32

	// Marker for destruction of inferior Organisms
	ToEliminate bool
	// Marks the species champion
	IsChampion bool

	// Number of reserved offspring for a population leader
	SuperChampOffspring int
	// Marks the best in population
	IsPopulationChampion bool
	// Marks the duplicate child of a champion (for tracking purposes)
	IsPopulationChampionChild bool

	// DEBUG variable - highest fitness of champ
	HighestFitness float64

	// Track its origin - for debugging or analysis - we can tell how the organism was born
	MutationStructBaby bool
	MateBaby bool
}

// Creates new organism with specified genome, fitness and given generation number
func NewOrganism(fit float64, g *Genome, gen int32) *Organism {
	return &Organism{
		Fitness:fit,
		GNome:g,
		Net:g.Genesis(g.GenomeId),
		Generation:gen,
	}
}

// Regenerate the network based on a change in the genotype
func (o *Organism) UpdatePhenotype() {
	// First, delete the old phenotype (net)
	o.Net = nil

	// Now, recreate the phenotype off the new genotype
	o.Net = o.GNome.Genesis(o.GNome.GenomeId)
}
