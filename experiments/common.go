// The experiments package holds various experiments with NEAT.
//
// The XOR experiment serves to actually check that network topology actually evolves and everything works as expected.
// Because XOR is not linearly separable, a neural network requires hidden units to solve it. The two inputs must be
// combined at some hidden unit, as opposed to only at the out- put node, because there is no function over a linear
// combination of the inputs that can separate the inputs into the proper classes. These structural requirements make
// XOR suitable for testing NEAT’s ability to evolve structure.
package experiments

import (
	"time"
	"github.com/yaricom/goNEAT/neat/genetics"
	"sort"
)

// Floats provides descriptive statistics on a slice of float64 values
type Floats []float64

// Max returns the greatest value in the slice
func (x Floats) Max() float64 {
	if len(x) == 0 {
		return 0.0
	}
	sort.Float64s(x)
	return x[len(x) - 1]
}

// The structure to represent one epoch execution results
type Epoch struct {
	// The generation ID for this epoch
	Id        int
	// The time when epoch was evaluated
	Executed  time.Time
	// The winner organism if was solved or nil
	Best      genetics.Organism
	// The flag to indicate whether experiment was solved in this epoch
	Solved    bool
	// The list of organisms fitness values in population
	Fitness   Floats
	// The novelty of organisms in population (less if much novel)
	Novelty   Floats
	// The list of organisms complexities in population
	Compexity Floats
}

// Epochs is a sortable collection of epochs
type Epochs []Epoch

func (is Epochs) Len() int {
	return len(is)
}
func (is Epochs) Swap(i, j int) {
	is[i], is[j] = is[j], is[i]
}
func (is Epochs) Less(i, j int) bool {
	if is[i].Executed.Equal(is[j].Executed) {
		return is[i].Id < is[j].Id // less is from earlier epochs
	}
	return is[i].Executed.Before(is[j].Executed) // less is from earlier time
}

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

func (t Trial) Best() genetics.Organism {
	var orgs = make([]genetics.Organism, 0, len(t.Epochs))
	for _, e := range t.Epochs {
		orgs = append(orgs, e.Best)
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

// Fitness returns the fitnesses of the best organism of each epoch
func (t Trial) Fitness() Floats {
	var x Floats = make([]float64, len(t.Epochs))
	for i, e := range t.Epochs {
		x[i] = e.Best.Fitness

	}
	return x
}

// Novelty returns the novelty values of the best organism of each epoch (less is much novel)
func (t Trial) Novelty() Floats {
	var x Floats = make([]float64, len(t.Epochs))
	for i, e := range t.Epochs {
		x[i] = e.Best.Generation
	}
	return x
}

// Complexity returns the complexity of the population
func (t Trial) Complexity() Floats {
	var x Floats = make([]float64, len(t.Epochs))
	for i, it := range t.Epochs {
		x[i] = float64(it.Best.Phenotype.Complexity())
	}
	return x
}

// Trials is a sortable collection of experiment runs (trials)
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

// The structure to hold statistics about conducted experiment
type Statistics struct {

}
