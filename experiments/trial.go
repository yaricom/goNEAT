package experiments

import (
	"time"
	"github.com/yaricom/goNEAT/neat/genetics"
	"sort"
	"encoding/gob"
)

// The structure to hold statistics about one experiment run (trial)
type Trial struct {
	// The trial number
	Id          int
	// The results per generation in this trial
	Generations Generations
}

func (t Trial) LastExecuted() time.Time {
	var u time.Time
	for _, i := range t.Generations {
		if u.Before(i.Executed) {
			u = i.Executed
		}
	}
	return u
}

func (t Trial) Best() *genetics.Organism {
	var orgs = make(genetics.Organisms, 0, len(t.Generations))
	for _, e := range t.Generations {
		orgs = append(orgs, e.Best)
	}
	sort.Sort(sort.Reverse(orgs))
	return orgs[0]
}

func (t Trial) Solved() bool {
	for _, e := range t.Generations {
		if e.Solved {
			return true
		}
	}
	return false
}

// Fitness returns the fitnesses of the best organism of each epoch in this trial
func (t Trial) Fitness() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		x[i] = e.Best.Fitness

	}
	return x
}

// Novelty returns the novelty values of the best species of each epoch
func (t Trial) Age() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		x[i] = float64(e.Best.Species.Age)
	}
	return x
}

// Complexity returns the complexity of the population
func (t Trial) Complexity() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		x[i] = float64(e.Best.Phenotype.Complexity())
	}
	return x
}

// Diversity returns number of species in each epoch
func (t Trial) Diversity() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		x[i] = float64(e.Diversity)
	}
	return x
}

// Returns number of nodes, genes and organism evaluations in the winner genome
func (t Trial) WinnerNGE() (nodes, genes, evals int) {
	for _, e := range t.Generations {
		if e.Solved {
			nodes = e.WinnerNodes
			genes = e.WinnerGenes
			evals = e.WinnerEvals
			break
		}
	}
	return nodes, genes, evals
}

// Encodes this trial
func (t *Trial) Encode(enc *gob.Encoder) error {
	err := enc.Encode(t.Id)
	err = enc.Encode(len(t.Generations))
	for _, e := range t.Generations {
		err = e.Encode(enc)
		if err != nil {
			return err
		}
	}
	return err
}

// Decodes trial data
func (t *Trial) Decode(dec *gob.Decoder) error {
	err := dec.Decode(&t.Id)
	var ngen int
	err = dec.Decode(&ngen)
	if err != nil {
		return err
	}
	t.Generations = make([]Generation, ngen)
	for i := 0; i < ngen; i++ {
		gen := Generation{}
		err = gen.Decode(dec)
		if err != nil {
			return err
		}
		t.Generations[i] = gen
	}
	return err
}

// Trials is a sortable collection of experiment runs (trials) by execution time and id
type Trials []Trial

func (ts Trials) Len() int {
	return len(ts)
}
func (ts Trials) Swap(i, j int) {
	ts[i], ts[j] = ts[j], ts[i]
}
func (ts Trials) Less(i, j int) bool {
	ui := ts[i].LastExecuted()
	uj := ts[j].LastExecuted()
	if ui.Equal(uj) {
		return ts[i].Id < ts[j].Id
	}
	return ui.Before(uj)
}
