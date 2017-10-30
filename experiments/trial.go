package experiments

import (
	"time"
	"github.com/yaricom/goNEAT/neat/genetics"
	"sort"
)

// The structure to hold statistics about one experiment run (trial)
type Trial struct {
	// The trial number
	Id     int
	// The trial epochs results
	Epochs Epochs
}

func (t Trial) LastExecuted() time.Time {
	var u time.Time
	for _, i := range t.Epochs {
		if u.Before(i.Executed) {
			u = i.Executed
		}
	}
	return u
}

func (t Trial) Best() *genetics.Organism {
	var orgs = make(genetics.Organisms, 0, len(t.Epochs))
	for _, e := range t.Epochs {
		orgs = append(orgs, &e.Best)
	}
	sort.Sort(sort.Reverse(orgs))
	return orgs[0]
}

func (t Trial) Solved() bool {
	for _, e := range t.Epochs {
		if e.Solved {
			return true
		}
	}
	return false
}

// Fitness returns the fitnesses of the best organism of each epoch in this trial
func (t Trial) Fitness() Floats {
	var x Floats = make([]float64, len(t.Epochs))
	for i, e := range t.Epochs {
		x[i] = e.Best.Fitness

	}
	return x
}

// Novelty returns the novelty values of the best species of each epoch
func (t Trial) Age() Floats {
	var x Floats = make([]float64, len(t.Epochs))
	for i, e := range t.Epochs {
		x[i] = float64(e.Best.Species.Age)
	}
	return x
}

// Complexity returns the complexity of the population
func (t Trial) Complexity() Floats {
	var x Floats = make([]float64, len(t.Epochs))
	for i, e := range t.Epochs {
		x[i] = float64(e.Best.Phenotype.Complexity())
	}
	return x
}

// Diversity returns number of species in each epoch
func (t Trial) Diversity() Floats {
	var x Floats = make([]float64, len(t.Epochs))
	for i, e := range t.Epochs {
		x[i] = float64(e.Diversity)
	}
	return x
}

// Returns average number of nodes in winner genomes among all epochs
func (t Trial) WinnerNodes() float64 {
	if len(t.Epochs) == 0 {
		return 0
	}
	total_nodes := 0
	for _, e := range t.Epochs {
		total_nodes += e.WinnerNodes
	}
	return float64(total_nodes) / float64(len(t.Epochs))
}

// Returns average number of genes in winner genomes among all epochs
func (t Trial) WinnerGenes() float64 {
	if len(t.Epochs) == 0 {
		return 0
	}
	total_genes := 0
	for _, e := range t.Epochs {
		total_genes += e.WinnerGenes
	}
	return float64(total_genes) / float64(len(t.Epochs))
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
