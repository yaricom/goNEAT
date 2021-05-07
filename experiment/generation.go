package experiment

import (
	"bytes"
	"encoding/gob"
	"github.com/pkg/errors"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"math"
	"reflect"
	"sort"
	"time"
)

// Generation the structure to represent execution results of one generation
type Generation struct {
	// The generation ID for this epoch
	Id int
	// The time when epoch was evaluated
	Executed time.Time
	// The elapsed time between generation execution start and finish
	Duration time.Duration
	// The best organism of best species
	Best *genetics.Organism
	// The flag to indicate whether experiment was solved in this epoch
	Solved bool

	// The list of organisms fitness values per species in population
	Fitness Floats
	// The age of organisms per species in population
	Age Floats
	// The list of organisms complexities per species in population
	Complexity Floats

	// The number of species in population at the end of this epoch
	Diversity int

	// The number of evaluations done before winner found
	WinnerEvals int
	// The number of nodes in winner genome or zero if not solved
	WinnerNodes int
	// The numbers of genes (links) in winner genome or zero if not solved
	WinnerGenes int

	// The ID of Trial this Generation was evaluated in
	TrialId int
}

// FillPopulationStatistics Collects statistics about given population
func (g *Generation) FillPopulationStatistics(pop *genetics.Population) {
	maxFitness := float64(math.MinInt64)
	g.Diversity = len(pop.Species)
	g.Age = make(Floats, g.Diversity)
	g.Complexity = make(Floats, g.Diversity)
	g.Fitness = make(Floats, g.Diversity)
	for i, currSpecies := range pop.Species {
		g.Age[i] = float64(currSpecies.Age)
		g.Complexity[i] = float64(currSpecies.Organisms[0].Phenotype.Complexity())
		g.Fitness[i] = currSpecies.Organisms[0].Fitness

		// find best organism in epoch if not solved
		if !g.Solved {
			// sort organisms from current species by fitness to have most fit first
			sort.Sort(sort.Reverse(currSpecies.Organisms))
			if currSpecies.Organisms[0].Fitness > maxFitness {
				maxFitness = currSpecies.Organisms[0].Fitness
				g.Best = currSpecies.Organisms[0]
			}
		}
	}
}

// Average Returns average fitness, age, and complexity among all organisms from population at the end of this epoch
func (g *Generation) Average() (fitness, age, complexity float64) {
	fitness = g.Fitness.Mean()
	age = g.Age.Mean()
	complexity = g.Complexity.Mean()
	return fitness, age, complexity
}

// Encode Encodes generation with provided GOB encoder
func (g *Generation) Encode(enc *gob.Encoder) error {
	if err := enc.EncodeValue(reflect.ValueOf(g.Id)); err != nil {
		return err
	}
	if err := enc.EncodeValue(reflect.ValueOf(g.Executed)); err != nil {
		return err
	}
	if err := enc.EncodeValue(reflect.ValueOf(g.Solved)); err != nil {
		return err
	}
	if err := enc.EncodeValue(reflect.ValueOf(g.Fitness)); err != nil {
		return err
	}
	if err := enc.EncodeValue(reflect.ValueOf(g.Age)); err != nil {
		return err
	}
	if err := enc.EncodeValue(reflect.ValueOf(g.Complexity)); err != nil {
		return err
	}
	if err := enc.EncodeValue(reflect.ValueOf(g.Diversity)); err != nil {
		return err
	}
	if err := enc.EncodeValue(reflect.ValueOf(g.WinnerEvals)); err != nil {
		return err
	}
	if err := enc.EncodeValue(reflect.ValueOf(g.WinnerNodes)); err != nil {
		return err
	}
	if err := enc.EncodeValue(reflect.ValueOf(g.WinnerGenes)); err != nil {
		return err
	}

	// encode best organism
	if g.Best != nil {
		if err := encodeOrganism(enc, g.Best); err != nil {
			return err
		}
	}
	return nil
}

func encodeOrganism(enc *gob.Encoder, org *genetics.Organism) error {
	if err := enc.Encode(org.Fitness); err != nil {
		return err
	}
	if err := enc.Encode(org.IsWinner); err != nil {
		return err
	}
	if err := enc.Encode(org.Generation); err != nil {
		return err
	}
	if err := enc.Encode(org.ExpectedOffspring); err != nil {
		return err
	}
	if err := enc.Encode(org.Error); err != nil {
		return err
	}

	// encode organism genome
	if org.Genotype != nil {
		if err := enc.Encode(org.Genotype.Id); err != nil {
			return err
		}
		outBuf := bytes.NewBufferString("")
		if err := org.Genotype.Write(outBuf); err != nil {
			return err
		}
		if err := enc.Encode(outBuf.Bytes()); err != nil {
			return err
		}
	}

	return nil
}

func (g *Generation) Decode(dec *gob.Decoder) error {
	if err := dec.Decode(&g.Id); err != nil {
		return errors.Wrap(err, "failed to decode Id")
	}
	if err := dec.Decode(&g.Executed); err != nil {
		return errors.Wrap(err, "failed to decode Executed")
	}
	if err := dec.Decode(&g.Solved); err != nil {
		return errors.Wrap(err, "failed to decode Solved")
	}
	if err := dec.Decode(&g.Fitness); err != nil {
		return errors.Wrap(err, "failed to decode Fitness")
	}
	if err := dec.Decode(&g.Age); err != nil {
		return errors.Wrap(err, "failed to decode Age")
	}
	if err := dec.Decode(&g.Complexity); err != nil {
		return errors.Wrap(err, "failed to decode Complexity")
	}
	if err := dec.Decode(&g.Diversity); err != nil {
		return errors.Wrap(err, "failed to decode Diversity")
	}
	if err := dec.Decode(&g.WinnerEvals); err != nil {
		return errors.Wrap(err, "failed to decode WinnerEvals")
	}
	if err := dec.Decode(&g.WinnerNodes); err != nil {
		return errors.Wrap(err, "failed to decode WinnerNodes")
	}
	if err := dec.Decode(&g.WinnerGenes); err != nil {
		return errors.Wrap(err, "failed to decode WinnerNodes")
	}

	// decode organism
	if org, err := decodeOrganism(dec); err != nil {
		return err
	} else {
		g.Best = org
	}
	return nil
}

func decodeOrganism(dec *gob.Decoder) (*genetics.Organism, error) {
	org := genetics.Organism{}
	if err := dec.Decode(&org.Fitness); err != nil {
		return nil, errors.Wrap(err, "failed to decode Fitness")
	}
	if err := dec.Decode(&org.IsWinner); err != nil {
		return nil, errors.Wrap(err, "failed to decode IsWinner")
	}
	if err := dec.Decode(&org.Generation); err != nil {
		return nil, errors.Wrap(err, "failed to decode Generation")
	}
	if err := dec.Decode(&org.ExpectedOffspring); err != nil {
		return nil, errors.Wrap(err, "failed to decode Generation")
	}
	if err := dec.Decode(&org.Error); err != nil {
		return nil, errors.Wrap(err, "failed to decode Generation")
	}

	// decode organism genome
	var genId int
	if err := dec.Decode(&genId); err != nil {
		return nil, errors.Wrap(err, "failed to decode genId")
	}
	var data []byte
	if err := dec.Decode(&data); err != nil {
		return nil, errors.Wrap(err, "failed to decode organism's data")
	}
	if gen, err := genetics.ReadGenome(bytes.NewBuffer(data), genId); err != nil {
		return nil, err
	} else {
		org.Genotype = gen
	}
	return &org, nil
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
