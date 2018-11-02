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
	Id               int
	// The results per generation in this trial
	Generations      Generations
	// The winner generation
	WinnerGeneration *Generation

	// The elapsed time between trial start and finish
	Duration         time.Duration
}

// Calculates average duration of evaluations among all generations of organism populations in this trial
func (t *Trial) AvgEpochDuration() time.Duration {
	total := time.Duration(0)
	for _, i := range t.Generations {
		total += i.Duration
	}
	return total / time.Duration(len(t.Generations))
}

func (t *Trial) LastExecuted() time.Time {
	var u time.Time
	for _, i := range t.Generations {
		if u.Before(i.Executed) {
			u = i.Executed
		}
	}
	return u
}

// Finds the most fit organism among all epochs in this trial. It's also possible to get the best organism only among the ones
// which was able to solve the experiment's problem.
func (t *Trial) BestOrganism(onlySolvers bool) (*genetics.Organism, bool) {
	var orgs = make(genetics.Organisms, 0, len(t.Generations))
	for _, e := range t.Generations {
		if !onlySolvers {
			// include all the most fit in each epoch
			orgs = append(orgs, e.Best)
		} else if e.Solved {
			// include only task solvers
			orgs = append(orgs, e.Best)
		}
	}
	if len(orgs) > 0 {
		sort.Sort(sort.Reverse(orgs))
		return orgs[0], true
	} else {
		return nil, false
	}
}

func (t *Trial) Solved() bool {
	for _, e := range t.Generations {
		if e.Solved {
			return true
		}
	}
	return false
}

// Fitness returns the fitnesses of the best organisms for each epoch in this trial
func (t *Trial) BestFitness() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		x[i] = e.Best.Fitness

	}
	return x
}

// Age returns the age of the best species for each epoch in this trial
func (t *Trial) BestAge() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		x[i] = float64(e.Best.Species.Age)
	}
	return x
}

// Complexity returns the complexity of the best species for each epoch in this trial
func (t *Trial) BestComplexity() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		x[i] = float64(e.Best.Phenotype.Complexity())
	}
	return x
}


// Diversity returns number of species for each epoch
func (t *Trial) Diversity() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		x[i] = float64(e.Diversity)
	}
	return x
}

// Returns average fitness, age, and complexity of population of organisms for each epoch in this trial
func (t *Trial) Average() (fitness, age, complexity Floats) {
	fitness = make(Floats, len(t.Generations))
	age = make(Floats, len(t.Generations))
	complexity = make(Floats, len(t.Generations))
	for i, e := range t.Generations {
		fitness[i], age[i], complexity[i] = e.Average()
	}
	return fitness, age, complexity
}

// Returns number of nodes, genes,  organism evaluations and species diversity in the winner genome
func (t *Trial) Winner() (nodes, genes, evals, diversity int) {
	if t.WinnerGeneration != nil {
		nodes = t.WinnerGeneration.WinnerNodes
		genes = t.WinnerGeneration.WinnerGenes
		evals = t.WinnerGeneration.WinnerEvals
		diversity = t.WinnerGeneration.Diversity
	} else {
		for _, e := range t.Generations {
			if e.Solved {
				nodes = e.WinnerNodes
				genes = e.WinnerGenes
				evals = e.WinnerEvals
				diversity = e.Diversity
				// Store winner
				t.WinnerGeneration = &e
				break
			}
		}
	}
	return nodes, genes, evals, diversity
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
