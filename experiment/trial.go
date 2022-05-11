package experiment

import (
	"encoding/gob"
	"github.com/yaricom/goNEAT/v3/neat/genetics"
	"sort"
	"time"
)

// Trial The structure to hold statistics about one experiment run (trial)
type Trial struct {
	// The trial number
	Id int
	// The results per generation in this trial
	Generations Generations
	// The winner generation
	WinnerGeneration *Generation

	// The elapsed time between trial start and finish
	Duration time.Duration
}

// AvgEpochDuration Calculates average duration of evaluations among all generations of organism populations in this trial
func (t *Trial) AvgEpochDuration() time.Duration {
	total := time.Duration(0)
	for _, i := range t.Generations {
		total += i.Duration
	}
	if len(t.Generations) > 0 {
		return total / time.Duration(len(t.Generations))
	} else {
		return EmptyDuration
	}
}

// RecentEpochEvalTime is to get time of the epoch executed most recently within this trial
func (t *Trial) RecentEpochEvalTime() time.Time {
	var u time.Time
	for _, i := range t.Generations {
		if u.Before(i.Executed) {
			u = i.Executed
		}
	}
	return u
}

// BestOrganism Finds the most fit organism among all epochs in this trial.
// It's also possible to get the best organism only among successful solvers of the experiment's problem.
func (t *Trial) BestOrganism(onlySolvers bool) (*genetics.Organism, bool) {
	var orgs = make(genetics.Organisms, 0, len(t.Generations))
	for _, e := range t.Generations {
		if !onlySolvers {
			// include every champion in each epoch
			orgs = append(orgs, e.Champion)
		} else if e.Solved {
			// include only successful task solver champions
			orgs = append(orgs, e.Champion)
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

// ChampionFitness returns the fitness values of the champion organisms for each epoch in this trial
func (t *Trial) ChampionFitness() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		if e.Champion != nil {
			x[i] = e.Champion.Fitness
		}
	}
	return x
}

// ChampionSpeciesAge returns the age of the species of the champion for each epoch in this trial
func (t *Trial) ChampionSpeciesAge() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		if e.Champion != nil && e.Champion.Species != nil {
			x[i] = float64(e.Champion.Species.Age)
		}
	}
	return x
}

// ChampionComplexity returns the complexity of the champion organism for each epoch in this trial
func (t *Trial) ChampionComplexity() Floats {
	var x Floats = make([]float64, len(t.Generations))
	for i, e := range t.Generations {
		if e.Champion != nil && e.Champion.Phenotype != nil {
			x[i] = float64(e.Champion.Phenotype.Complexity())
		}
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

// Average the average fitness, age, and complexity of the best organisms per species for each epoch in this trial
func (t *Trial) Average() (fitness, age, complexity Floats) {
	fitness = make(Floats, len(t.Generations))
	age = make(Floats, len(t.Generations))
	complexity = make(Floats, len(t.Generations))
	for i, e := range t.Generations {
		fitness[i], age[i], complexity[i] = e.Average()
	}
	return fitness, age, complexity
}

// Winner the number of nodes, genes, organism evaluations and species diversity in the winner genome
func (t *Trial) Winner() (nodes, genes, evals, diversity int) {
	if t.WinnerGeneration != nil {
		nodes = t.WinnerGeneration.WinnerNodes
		genes = t.WinnerGeneration.WinnerGenes
		evals = t.WinnerGeneration.WinnerEvals
		diversity = t.WinnerGeneration.Diversity
	} else if len(t.Generations) > 0 {
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
	} else {
		nodes, genes, evals, diversity = -1, -1, -1, -1
	}
	return nodes, genes, evals, diversity
}

// Encode is to encode this trial
func (t *Trial) Encode(enc *gob.Encoder) error {
	if err := enc.Encode(t.Id); err != nil {
		return err
	}
	if err := enc.Encode(len(t.Generations)); err != nil {
		return err
	}
	for _, e := range t.Generations {
		if err := e.Encode(enc); err != nil {
			return err
		}
	}
	return nil
}

// Decode Decodes trial data
func (t *Trial) Decode(dec *gob.Decoder) error {
	if err := dec.Decode(&t.Id); err != nil {
		return err
	}
	var ngen int
	if err := dec.Decode(&ngen); err != nil {
		return err
	}
	t.Generations = make([]Generation, ngen)
	for i := 0; i < ngen; i++ {
		gen := Generation{}
		if err := gen.Decode(dec); err != nil {
			return err
		}
		t.Generations[i] = gen
	}
	return nil
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
	ui := ts[i].RecentEpochEvalTime()
	uj := ts[j].RecentEpochEvalTime()
	if ui.Equal(uj) {
		return ts[i].Id < ts[j].Id
	}
	return ui.Before(uj)
}
