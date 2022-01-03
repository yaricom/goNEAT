package genetics

import (
	"bytes"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat/network"
)

// Organisms represents sortable list of organisms by fitness
type Organisms []*Organism

// OrganismData is the object to associate implementation specific data with particular organism for various algorithm implementations
type OrganismData struct {
	// The implementation specific data object to be associated with organism
	Value interface{}
}

// Organism is Genotypes (Genomes) and Phenotypes (Networks) with fitness information,
// i.e. the genotype and phenotype together.
type Organism struct {
	// A measure of fitness for the Organism
	Fitness float64
	// The error value indicating how far organism's performance is from ideal task goal, e.g. MSE
	Error float64
	// Win marker (if needed for a particular task)
	IsWinner bool

	// The Organism's phenotype
	Phenotype *network.Network
	// The Organism's genotype
	Genotype *Genome
	// The Species of the Organism
	Species *Species

	// Number of children this Organism may have
	ExpectedOffspring float64
	// Tells which generation this Organism is from
	Generation int

	// The utility data transfer object to be used by different GA implementations to hold additional data.
	// Implemented as ANY to allow implementation specific objects.
	Data *OrganismData

	// A fitness measure that won't change during fitness adjustments of population's epoch evaluation
	originalFitness float64

	// Marker for destruction of inferior Organisms
	toEliminate bool
	// Marks the species champion
	isChampion bool

	// Number of reserved offspring for a population leader
	superChampOffspring int
	// Marks the best in population
	isPopulationChampion bool
	// Marks the duplicate child of a champion (for tracking purposes)
	isPopulationChampionChild bool

	// DEBUG variable - highest fitness of champ
	highestFitness float64

	// Track its origin - for debugging or analysis - we can tell how the organism was born
	mutationStructBaby bool
	mateBaby           bool

	// The flag to be used as utility value
	Flag int
}

// NewOrganism Creates new organism with specified genome, fitness and given generation number
func NewOrganism(fit float64, g *Genome, generation int) (org *Organism, err error) {
	phenotype := g.Phenotype
	if phenotype == nil {
		phenotype, err = g.Genesis(g.Id)
		if err != nil {
			return nil, err
		}
	}
	org = &Organism{
		Fitness:    fit,
		Genotype:   g,
		Phenotype:  phenotype,
		Generation: generation,
	}
	return org, nil
}

// UpdatePhenotype Regenerate the underlying network graph based on a change in the genotype
func (o *Organism) UpdatePhenotype() (err error) {
	// First, delete the old phenotype (net)
	o.Phenotype = nil

	// Now, recreate the phenotype off the new genotype
	o.Phenotype, err = o.Genotype.Genesis(o.Genotype.Id)
	return err
}

// CheckChampionChildDamaged Method to check if this organism is a child of the champion
// but has the fitness score less than of the parent. This can be used to check if champion's offsprings degraded.
func (o *Organism) CheckChampionChildDamaged() bool {
	if o.isPopulationChampionChild && o.highestFitness > o.Fitness {
		return true
	}
	return false
}

// MarshalBinary Encodes this organism for wired transmission during parallel reproduction cycle
func (o *Organism) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	if _, err := fmt.Fprintln(&buf, o.Fitness, o.Generation, o.highestFitness, o.isPopulationChampionChild, o.Genotype.Id); err != nil {
		return nil, err
	} else if err = o.Genotype.Write(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// UnmarshalBinary Decodes organism received over the wire during parallel reproduction cycle
func (o *Organism) UnmarshalBinary(data []byte) error {
	// A simple encoding: plain text.
	b := bytes.NewBuffer(data)
	var genotypeId int
	if _, err := fmt.Fscanln(b, &o.Fitness, &o.Generation, &o.highestFitness, &o.isPopulationChampionChild, &genotypeId); err != nil {
		return err
	} else if o.Genotype, err = ReadGenome(b, genotypeId); err != nil {
		return err
	} else if o.Phenotype, err = o.Genotype.Genesis(genotypeId); err != nil {
		return err
	}

	return nil
}

func (o *Organism) String() string {
	champStr := ""
	if o.isChampion {
		champStr = " - CHAMPION - "
	}
	eliminateStr := ""
	if o.toEliminate {
		eliminateStr = " - TO BE ELIMINATED - "
	}
	return fmt.Sprintf("[Organism generation: %d, fitness: %.3f, original fitness: %.3f%s%s]",
		o.Generation, o.Fitness, o.originalFitness, champStr, eliminateStr)
}

// Dump is to dump all organism's fields into string
func (o *Organism) Dump() string {
	b := bytes.NewBufferString("Organism:")
	_, _ = fmt.Fprintln(b, "Fitness: ", o.Fitness)
	_, _ = fmt.Fprintln(b, "Error: ", o.Error)
	_, _ = fmt.Fprintln(b, "IsWinner: ", o.IsWinner)
	_, _ = fmt.Fprintln(b, "Phenotype: ", o.Phenotype)
	_, _ = fmt.Fprintln(b, "Genotype: ", o.Genotype)
	_, _ = fmt.Fprintln(b, "Species: ", o.Species)
	_, _ = fmt.Fprintln(b, "ExpectedOffspring: ", o.ExpectedOffspring)
	_, _ = fmt.Fprintln(b, "Data: ", o.Data)
	_, _ = fmt.Fprintln(b, "Phenotype: ", o.Phenotype)
	_, _ = fmt.Fprintln(b, "originalFitness: ", o.originalFitness)
	_, _ = fmt.Fprintln(b, "toEliminate: ", o.toEliminate)
	_, _ = fmt.Fprintln(b, "isChampion: ", o.isChampion)
	_, _ = fmt.Fprintln(b, "superChampOffspring: ", o.superChampOffspring)
	_, _ = fmt.Fprintln(b, "isPopulationChampion: ", o.isPopulationChampion)
	_, _ = fmt.Fprintln(b, "isPopulationChampionChild: ", o.isPopulationChampionChild)
	_, _ = fmt.Fprintln(b, "highestFitness: ", o.highestFitness)
	_, _ = fmt.Fprintln(b, "mutationStructBaby: ", o.mutationStructBaby)
	_, _ = fmt.Fprintln(b, "mateBaby: ", o.mateBaby)
	_, _ = fmt.Fprintln(b, "Flag: ", o.Flag)

	return b.String()
}

// The Organisms sort interface implementation

func (f Organisms) Len() int {
	return len(f)
}
func (f Organisms) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}
func (f Organisms) Less(i, j int) bool {
	if f[i].Fitness < f[j].Fitness {
		// try to promote most fit organisms
		return true // lower fitness is less
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
