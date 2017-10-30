package experiments

import (
	"time"
	"sort"
	"github.com/yaricom/goNEAT/neat/genetics"
)

// An Experiment is a collection of trials for one experiment. It's useful for statistical analysis of a series of
// experiments
type Experiment struct {
	Id          int
	Name string
	Trials
}

func (e Experiment) LastExecuted() time.Time {
	var u time.Time
	for _, e := range e.Trials {
		ut := e.LastExecuted()
		if u.Before(ut) {
			u = ut
		}
	}
	return u
}

func (e Experiment) Best() *genetics.Organism {
	var orgs = make(genetics.Organisms, len(e.Trials))
	for i, t := range e.Trials {
		orgs[i] = t.Best()
	}
	sort.Sort(sort.Reverse(orgs))
	return orgs[0]
}

func (e Experiment) Solved() bool {
	for _, t := range e.Trials {
		if t.Solved() {
			return true
		}
	}
	return false
}

// Fitness returns the fitnesses of the best genome of each trial
func (e Experiment) Fitness() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		x[i] = t.Best().Fitness
	}
	return x
}

// Novelty returns the age values of the best species of each trial
func (e Experiment) Age() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		x[i] = float64(t.Best().Species.Age)
	}
	return x
}

// Complexity returns the complexity of the best genome of each trial
func (e Experiment) Complexity() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		x[i] = float64(t.Best().Phenotype.Complexity())
	}
	return x
}

// Diversity returns the average number of species in each trial
func (e Experiment) Diversity() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		x[i] = t.Diversity().Mean()
	}
	return x
}

// Trials returns the number of epochs in each trial
func (e Experiment) Epochs() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		x[i] = float64(len(t.Epochs))
	}
	return x
}

// Experiments is a sortable list of experiments by execution time and Id
type Experiments []Experiment

func (es Experiments) Len() int      { return len(es) }
func (es Experiments) Swap(i, j int) { es[i], es[j] = es[j], es[i] }
func (es Experiments) Less(i, j int) bool {
	ui := es[i].LastExecuted()
	uj := es[j].LastExecuted()
	if ui.Equal(uj) {
		return es[i].Id < es[j].Id
	}
	return ui.Before(uj)
}