package experiments

import (
	"time"
	"github.com/yaricom/goNEAT/neat/genetics"
	"math"
	"encoding/gob"
	"bytes"
	"reflect"
	"sort"
)

// The structure to represent execution results of one generation
type Generation struct {
	// The generation ID for this epoch
	Id          int
	// The time when epoch was evaluated
	Executed    time.Time
	// The elapsed time between generation execution start and finish
	Duration    time.Duration
	// The best organism of best species
	Best        *genetics.Organism
	// The flag to indicate whether experiment was solved in this epoch
	Solved      bool

	// The list of organisms fitness values per species in population
	Fitness     Floats
	// The age of organisms per species in population
	Age         Floats
	// The list of organisms complexities per species in population
	Compexity   Floats

	// The number of species in population at the end of this epoch
	Diversity   int

	// The number of evaluations done before winner found
	WinnerEvals int
	// The number of nodes in winner genome or zero if not solved
	WinnerNodes int
	// The numbers of genes (links) in winner genome or zero if not solved
	WinnerGenes int

	// The ID of Trial this Generation was evaluated in
	TrialId     int
}

// Collects statistics about given population
func (epoch *Generation) FillPopulationStatistics(pop *genetics.Population) {
	max_fitness := float64(math.MinInt64)
	epoch.Diversity = len(pop.Species)
	epoch.Age = make(Floats, epoch.Diversity)
	epoch.Compexity = make(Floats, epoch.Diversity)
	epoch.Fitness = make(Floats, epoch.Diversity)
	for i, curr_species := range pop.Species {
		epoch.Age[i] = float64(curr_species.Age)
		epoch.Compexity[i] = float64(curr_species.Organisms[0].Phenotype.Complexity())
		epoch.Fitness[i] = curr_species.Organisms[0].Fitness

		// find best organism in epoch if not solved
		if !epoch.Solved {
			// sort organisms from current species by fitness to have most fit first
			sort.Sort(sort.Reverse(curr_species.Organisms))
			if curr_species.Organisms[0].Fitness > max_fitness {
				max_fitness = curr_species.Organisms[0].Fitness
				epoch.Best = curr_species.Organisms[0]
			}
		}
	}
}

// Returns average fitness, age, and complexity among all organisms from population at the end of this epoch
func (epoch *Generation) Average() (fitness, age, complexity float64) {
	fitness = epoch.Fitness.Mean()
	age = epoch.Age.Mean()
	complexity = epoch.Compexity.Mean()
	return fitness, age, complexity
}

// Encodes generation with provided GOB encoder
func (epoch *Generation) Encode(enc *gob.Encoder) error {
	err := enc.EncodeValue(reflect.ValueOf(epoch.Id))
	err = enc.EncodeValue(reflect.ValueOf(epoch.Executed))
	err = enc.EncodeValue(reflect.ValueOf(epoch.Solved))
	err = enc.EncodeValue(reflect.ValueOf(epoch.Fitness))
	err = enc.EncodeValue(reflect.ValueOf(epoch.Age))
	err = enc.EncodeValue(reflect.ValueOf(epoch.Compexity))
	err = enc.EncodeValue(reflect.ValueOf(epoch.Diversity))
	err = enc.EncodeValue(reflect.ValueOf(epoch.WinnerEvals))
	err = enc.EncodeValue(reflect.ValueOf(epoch.WinnerNodes))
	err = enc.EncodeValue(reflect.ValueOf(epoch.WinnerGenes))

	if err != nil {
		return err
	}

	// encode best organism
	if epoch.Best != nil {
		err = encodeOrganism(enc, epoch.Best)
	}
	return err
}

func encodeOrganism(enc *gob.Encoder, org *genetics.Organism) error {
	err := enc.Encode(org.Fitness)
	err = enc.Encode(org.IsWinner)
	err = enc.Encode(org.Generation)
	err = enc.Encode(org.ExpectedOffspring)
	err = enc.Encode(org.Error)

	if err != nil {
		return err
	}

	// encode organism genome
	if org.Genotype != nil {
		err = enc.Encode(org.Genotype.Id)
		out_buf := bytes.NewBufferString("")
		org.Genotype.Write(out_buf)
		err = enc.Encode(out_buf.Bytes())
	}

	return err
}

func (epoch *Generation) Decode(dec *gob.Decoder) error {
	err := dec.Decode(&epoch.Id)
	err = dec.Decode(&epoch.Executed)
	err = dec.Decode(&epoch.Solved)
	err = dec.Decode(&epoch.Fitness)
	err = dec.Decode(&epoch.Age)
	err = dec.Decode(&epoch.Compexity)
	err = dec.Decode(&epoch.Diversity)
	err = dec.Decode(&epoch.WinnerEvals)
	err = dec.Decode(&epoch.WinnerNodes)
	err = dec.Decode(&epoch.WinnerGenes)

	if err != nil {
		return err
	}

	// decode organism
	org, err := decodeOrganism(dec)
	if err == nil {
		epoch.Best = org
	}
	return err
}

func decodeOrganism(dec *gob.Decoder) (*genetics.Organism, error) {
	org := genetics.Organism{}
	err := dec.Decode(&org.Fitness)
	err = dec.Decode(&org.IsWinner)
	err = dec.Decode(&org.Generation)
	err = dec.Decode(&org.ExpectedOffspring)
	err = dec.Decode(&org.Error)

	if err != nil {
		return nil, err
	}

	// decode organism genome
	var gen_id int
	err = dec.Decode(&gen_id)
	var data []byte
	err = dec.Decode(&data)
	gen, err := genetics.ReadGenome(bytes.NewBuffer(data), gen_id)
	org.Genotype = gen

	return &org, err
}

// Generations is a sortable collection of generations by execution time and Id
type Generations []Generation

func (is Generations) Len() int {
	return len(is)
}
func (is Generations) Swap(i, j int) {
	is[i], is[j] = is[j], is[i]
}
func (is Generations) Less(i, j int) bool {
	if is[i].Executed.Equal(is[j].Executed) {
		return is[i].Id < is[j].Id // less is from earlier epochs
	}
	return is[i].Executed.Before(is[j].Executed) // less is from earlier time
}
