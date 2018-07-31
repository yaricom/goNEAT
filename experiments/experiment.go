package experiments

import (
	"time"
	"sort"
	"github.com/yaricom/goNEAT/neat/genetics"
	"io"
	"encoding/gob"
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
		x[i] = float64(len(t.Generations))
	}
	return x
}

// Returns average number of nodes, genes and organisms evaluations of winner genomes among all trials, i.e. for all trials
// where winning solution was found
func (e Experiment) AvgWinnerNGE() (avg_nodes, avg_genes, avg_evals float64) {
	total_nodes, total_genes, total_evals := 0, 0, 0
	count := 0
	for _, t := range e.Trials {
		if t.Solved() {
			nodes, genes, evals := t.WinnerNGE()
			total_nodes += nodes
			total_genes += genes
			total_evals += evals

			count++
		}
	}
	avg_nodes = float64(total_nodes) / float64(count)
	avg_genes = float64(total_genes) / float64(count)
	avg_evals = float64(total_evals) / float64(count)
	return avg_nodes, avg_genes, avg_evals
}

// Encodes experiment and writes to provided writer
func (ex *Experiment) Write(w io.Writer) error {
	enc := gob.NewEncoder(w)
	return ex.Encode(enc)
}

// Encodes experiment with GOB encoding
func (ex *Experiment) Encode(enc *gob.Encoder) error {
	err := enc.Encode(ex.Id)
	err = enc.Encode(ex.Name)

	// encode trials
	err = enc.Encode(len(ex.Trials))
	for _, t := range ex.Trials {
		err = t.Encode(enc)
		if err != nil {
			return err
		}
	}
	return err
}

// Reads experiment data from provided reader and decodes it
func (ex *Experiment) Read(r io.Reader) error {
	dec := gob.NewDecoder(r)
	return ex.Decode(dec)
}

// Decodes experiment data
func (ex *Experiment) Decode(dec *gob.Decoder) error {
	err := dec.Decode(&ex.Id)
	err = dec.Decode(&ex.Name)

	// decode Trials
	var t_num int
	err = dec.Decode(&t_num)
	if err != nil {
		return err
	}

	ex.Trials = make([]Trial, t_num)
	for i := 0; i< t_num; i++ {
		trial := Trial{}
		err = trial.Decode(dec)
		ex.Trials[i] = trial
	}
	return err
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