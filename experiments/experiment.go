package experiments

import (
	"time"
	"sort"
	"github.com/yaricom/goNEAT/neat/genetics"
	"io"
	"encoding/gob"
	"fmt"
)

// An Experiment is a collection of trials for one experiment. It's useful for statistical analysis of a series of
// experiments
type Experiment struct {
	Id   int
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

// Finds the most fit organism among all epochs in this trial. It's also possible to get the best organism only among the ones
// which was able to solve the experiment's problem. Returns the best fit organism in this experiment among with ID of trial
// where it was found and boolean value to indicate if search was successful.
func (e Experiment) BestOrganism(onlySolvers bool) (*genetics.Organism, int, bool) {
	var orgs = make(genetics.Organisms, 0, len(e.Trials))
	for i, t := range e.Trials {
		org, found := t.BestOrganism(onlySolvers)
		if found {
			orgs = append(orgs, org)
			org.Flag = i
		}

	}
	if len(orgs) > 0 {
		sort.Sort(sort.Reverse(orgs))
		return orgs[0], orgs[0].Flag, true
	} else {
		return nil, -1, false
	}

}

func (e Experiment) Solved() bool {
	for _, t := range e.Trials {
		if t.Solved() {
			return true
		}
	}
	return false
}

// The fitness values of the best organisms for each trial
func (e Experiment) BestFitness() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		if org, ok := t.BestOrganism(false); ok {
			x[i] = org.Fitness
		}
	}
	return x
}

// The age values of the organisms for each trial
func (e Experiment) BestAge() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		if org, ok := t.BestOrganism(false); ok {
			x[i] = float64(org.Species.Age)
		}
	}
	return x
}

// The complexity values of the best organisms for each trial
func (e Experiment) BestComplexity() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		if org, ok := t.BestOrganism(false); ok {
			x[i] = float64(org.Phenotype.Complexity())
		}
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

// Returns average number of nodes, genes, organisms evaluations, and species diversity of winner genomes among all
// trials, i.e. for all trials where winning solution was found
func (e Experiment) AvgWinner() (avg_nodes, avg_genes, avg_evals, avg_diversity float64) {
	total_nodes, total_genes, total_evals, total_diversity := 0, 0, 0, 0
	count := 0
	for _, t := range e.Trials {
		if t.Solved() {
			nodes, genes, evals, diversity := t.Winner()
			total_nodes += nodes
			total_genes += genes
			total_evals += evals
			total_diversity += diversity

			count++
		}
	}
	avg_nodes = float64(total_nodes) / float64(count)
	avg_genes = float64(total_genes) / float64(count)
	avg_evals = float64(total_evals) / float64(count)
	avg_diversity = float64(total_diversity) / float64(count)
	return avg_nodes, avg_genes, avg_evals, avg_diversity
}

// Prints experiment statistics
func (ex Experiment) PrintStatistics() {
	// Print absolute champion statistics
	if org, trid, found := ex.BestOrganism(true); found {
		nodes, genes, evals, divers := ex.Trials[trid].Winner()
		fmt.Printf("\nChampion found in %d trial\n\tWinner Nodes:\t%.1f\n\tWinner Genes:\t%.1f\n\tWinner Evals:\t%.1f\n\tDiversity:\t%.1f\n",
			trid, nodes, genes, evals, divers)
		fmt.Printf("\tComplexity:\t%d\n\tAge:\t\t%d\n",
			org.Phenotype.Complexity(), org.Species.Age)
	} else {
		fmt.Println("\nNo winner found in the experiment!!!")
	}

	// Print average winner statistics
	if len(ex.Trials) > 1 {
		avg_nodes, avg_genes, avg_evals, avg_divers := ex.AvgWinner()

		fmt.Printf("\nAverage\n\tWinner Nodes:\t%.1f\n\tWinner Genes:\t%.1f\n\tWinner Evals:\t%.1f\n\tDiversity:\t%.1f\n",
			avg_nodes, avg_genes, avg_evals, avg_divers)
	}

	// Print best organisms statistics per epoch per trial, i.e. every the best found
	mean_complexity, mean_diversity, mean_age := 0.0, 0.0, 0.0
	for _, t := range ex.Trials {
		mean_complexity += t.BestComplexity().Mean()
		mean_diversity += t.Diversity().Mean()
		mean_age += t.BestAge().Mean()
	}
	count := float64(len(ex.Trials))
	mean_complexity /= count
	mean_diversity /= count
	mean_age /= count
	fmt.Printf("Mean of the all most fit organisms found\n\tComplexity:\t%.1f\n\tDiversity:\t%.1f\n\tAge:\t\t%.1f\n", mean_complexity, mean_diversity, mean_age)

}

// Encodes experiment and writes to provided writer
func (ex Experiment) Write(w io.Writer) error {
	enc := gob.NewEncoder(w)
	return ex.Encode(enc)
}

// Encodes experiment with GOB encoding
func (ex Experiment) Encode(enc *gob.Encoder) error {
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
	for i := 0; i < t_num; i++ {
		trial := Trial{}
		err = trial.Decode(dec)
		ex.Trials[i] = trial
	}
	return err
}

// Experiments is a sortable list of experiments by execution time and Id
type Experiments []Experiment

func (es Experiments) Len() int {
	return len(es)
}
func (es Experiments) Swap(i, j int) {
	es[i], es[j] = es[j], es[i]
}
func (es Experiments) Less(i, j int) bool {
	ui := es[i].LastExecuted()
	uj := es[j].LastExecuted()
	if ui.Equal(uj) {
		return es[i].Id < es[j].Id
	}
	return ui.Before(uj)
}