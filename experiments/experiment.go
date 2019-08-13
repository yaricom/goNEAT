package experiments

import (
	"time"
	"sort"
	"github.com/yaricom/goNEAT/neat/genetics"
	"io"
	"encoding/gob"
	"fmt"
	"math"
)

// An Experiment is a collection of trials for one experiment. It's useful for statistical analysis of a series of
// experiments
type Experiment struct {
	Id              int
	Name            string
	Trials
	// The maximal allowed fitness score as defined by fitness function of experiment.
	// It is used to normalize fitness score value used in efficiency score calculation. If this value
	// is not set, than fitness score will not be normalized during efficiency score estimation.
	MaxFintessScore float64
}

// Calculates average duration of experiment's trial
// Note, that most trials finish after solution solved, so this metric can be used to represent how efficient the solvers
// was generated
func (e *Experiment) AvgTrialDuration() time.Duration {
	total := time.Duration(0)
	for _, t := range e.Trials {
		total += t.Duration
	}
	return total / time.Duration(len(e.Trials))
}

// Calculates average duration of evaluations among all generations of organism populations in this experiment
func (e *Experiment) AvgEpochDuration() time.Duration {
	total := time.Duration(0)
	for _, t := range e.Trials {
		total += t.AvgEpochDuration()
	}
	return total / time.Duration(len(e.Trials))
}

// Calculates average number of generations evaluated per trial during this experiment. This can be helpful when estimating
// algorithm efficiency, because when winner organism is found the trial is terminated, i.e. less evaluations - more fast
// convergence.
func (e *Experiment) AvgGenerationsPerTrial() float64 {
	total := 0.0
	for _, t := range e.Trials {
		total += float64(len(t.Generations))
	}
	return total / float64(len(e.Trials))
}

// Returns time of last trial's execution
func (e *Experiment) LastExecuted() time.Time {
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
func (e *Experiment) BestOrganism(onlySolvers bool) (*genetics.Organism, int, bool) {
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

func (e *Experiment) Solved() bool {
	for _, t := range e.Trials {
		if t.Solved() {
			return true
		}
	}
	return false
}

// The fitness values of the best organisms for each trial
func (e *Experiment) BestFitness() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		if org, ok := t.BestOrganism(false); ok {
			x[i] = org.Fitness
		}
	}
	return x
}

// The age values of the organisms for each trial
func (e *Experiment) BestAge() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		if org, ok := t.BestOrganism(false); ok {
			x[i] = float64(org.Species.Age)
		}
	}
	return x
}

// The complexity values of the best organisms for each trial
func (e *Experiment) BestComplexity() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		if org, ok := t.BestOrganism(false); ok {
			x[i] = float64(org.Phenotype.Complexity())
		}
	}
	return x
}

// Diversity returns the average number of species in each trial
func (e *Experiment) Diversity() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		x[i] = t.Diversity().Mean()
	}
	return x
}

// Trials returns the number of epochs in each trial
func (e *Experiment) Epochs() Floats {
	var x Floats = make([]float64, len(e.Trials))
	for i, t := range e.Trials {
		x[i] = float64(len(t.Generations))
	}
	return x
}

// The number of trials solved
func (e *Experiment) TrialsSolved() int {
	count := 0
	for _, t := range e.Trials {
		if t.Solved() {
			count++
		}
	}
	return count
}

// The success rate
func (e *Experiment) SuccessRate() float64 {
	soved := float64(e.TrialsSolved())
	return soved / float64(len(e.Trials))
}

// Returns average number of nodes, genes, organisms evaluations, and species diversity of winner genomes among all
// trials, i.e. for all trials where winning solution was found
func (e *Experiment) AvgWinner() (avg_nodes, avg_genes, avg_evals, avg_diversity float64) {
	total_nodes, total_genes, total_evals, total_diversity := 0, 0, 0, 0
	count := 0
	for i := 0; i < len(e.Trials); i++ {
		t := e.Trials[i]
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

// Calculates the efficiency score of the solution
// We are interested in efficient solver search solution that take
// less time per epoch, less generations per trial, and produce less complicated winner genomes.
// At the same time it should have maximal fitness score and maximal success rate among trials.
func (ex *Experiment) EfficiencyScore() float64 {
	mean_complexity, mean_fitness := 0.0, 0.0
	if len(ex.Trials) > 1 {
		count := 0.0
		for i := 0; i < len(ex.Trials); i++ {
			t := ex.Trials[i]
			if t.Solved() {
				if t.WinnerGeneration == nil {
					// find winner
					t.Winner()
				}

				mean_complexity += float64(t.WinnerGeneration.Best.Phenotype.Complexity())
				mean_fitness += t.WinnerGeneration.Best.Fitness

				count++
			}
		}
		mean_complexity /= count
		mean_fitness /= count
	}

	// normalize and scale fitness score if appropriate
	fitness_score := mean_fitness
	if ex.MaxFintessScore > 0 {
		fitness_score /= ex.MaxFintessScore
		fitness_score *= 100
	}

	score := ex.AvgEpochDuration().Seconds() * 1000.0 * ex.AvgGenerationsPerTrial() * mean_complexity
	if score > 0 {
		score = ex.SuccessRate() * fitness_score / math.Log(score)
	}

	return score
}

// Prints experiment statistics
func (ex *Experiment) PrintStatistics() {
	fmt.Printf("\nSolved %d trials from %d, success rate: %f\n", ex.TrialsSolved(), len(ex.Trials), ex.SuccessRate())
	fmt.Printf("Average\n\tTrial duration:\t\t%s\n\tEpoch duration:\t\t%s\n\tGenerations/trial:\t%.1f\n",
		ex.AvgTrialDuration(), ex.AvgEpochDuration(), ex.AvgGenerationsPerTrial())
	// Print absolute champion statistics
	if org, trid, found := ex.BestOrganism(true); found {
		nodes, genes, evals, divers := ex.Trials[trid].Winner()
		fmt.Printf("\nChampion found in %d trial run\n\tWinner Nodes:\t\t%d\n\tWinner Genes:\t\t%d\n\tWinner Evals:\t\t%d\n\n\tDiversity:\t\t%d",
			trid, nodes, genes, evals, divers)
		fmt.Printf("\n\tComplexity:\t\t%d\n\tAge:\t\t\t%d\n\tFitness:\t\t%f\n",
			org.Phenotype.Complexity(), org.Species.Age, org.Fitness)
	} else {
		fmt.Println("\nNo winner found in the experiment!!!")
	}

	// Print average winner statistics
	mean_complexity, mean_diversity, mean_age, mean_fitness := 0.0, 0.0, 0.0, 0.0
	if len(ex.Trials) > 1 {
		avg_nodes, avg_genes, avg_evals, avg_divers, avg_generations := 0.0, 0.0, 0.0, 0.0, 0.0
		count := 0.0
		for i := 0; i < len(ex.Trials); i++ {
			t := ex.Trials[i]

			if t.Solved() {
				nodes, genes, evals, diversity := t.Winner()
				avg_nodes += float64(nodes)
				avg_genes += float64(genes)
				avg_evals += float64(evals)
				avg_divers += float64(diversity)
				avg_generations += float64(len(t.Generations))

				mean_complexity += float64(t.WinnerGeneration.Best.Phenotype.Complexity())
				mean_age += float64(t.WinnerGeneration.Best.Species.Age)
				mean_fitness += t.WinnerGeneration.Best.Fitness

				count++

				// update trials array
				ex.Trials[i] = t
			}
		}
		avg_nodes /= count
		avg_genes /= count
		avg_evals /= count
		avg_divers /= count
		avg_generations /= count
		fmt.Printf("\nAverage among winners\n\tWinner Nodes:\t\t%.1f\n\tWinner Genes:\t\t%.1f\n\tWinner Evals:\t\t%.1f\n\tGenerations/trial:\t%.1f\n\n\tDiversity:\t\t%f\n",
			avg_nodes, avg_genes, avg_evals, avg_generations, avg_divers)

		mean_complexity /= count
		mean_age /= count
		mean_fitness /= count
		fmt.Printf("\tComplexity:\t\t%f\n\tAge:\t\t\t%f\n\tFitness:\t\t%f\n",
			mean_complexity, mean_age, mean_fitness)
	}

	// Print the average values for each population of organisms evaluated
	count := float64(len(ex.Trials))
	for _, t := range ex.Trials {
		fitness, age, complexity := t.Average()

		mean_complexity += complexity.Mean()
		mean_diversity += t.Diversity().Mean()
		mean_age += age.Mean()
		mean_fitness += fitness.Mean()
	}
	mean_complexity /= count
	mean_diversity /= count
	mean_age /= count
	mean_fitness /= count
	fmt.Printf("\nAverages for all organisms evaluated during experiment\n\tDiversity:\t\t%f\n\tComplexity:\t\t%f\n\tAge:\t\t\t%f\n\tFitness:\t\t%f\n",
		mean_diversity, mean_complexity, mean_age, mean_fitness)

	score := ex.EfficiencyScore()
	fmt.Printf("\nEfficiency score:\t\t%f\n\n", score)
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