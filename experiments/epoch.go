package experiments

import (
	"time"
	"github.com/yaricom/goNEAT/neat/genetics"
)

// The structure to represent one epoch execution results
type Epoch struct {
	// The generation ID for this epoch
	Id          int
	// The time when epoch was evaluated
	Executed    time.Time
	// The best organism of best species
	Best        *genetics.Organism
	// The flag to indicate whether experiment was solved in this epoch
	Solved      bool
	// The list of organisms fitness values per species in population
	Fitness     Floats
	// The age of organisms per species in population
	Age         Floats
	// The list of organisms complexities per species in population
	Compexity   Floats
	// The number of species in population
	Diversity   int

	// The number of evaluations done before winner found
	WinnerEvals int
	// The number of nodes in winner genome or zero if not solved
	WinnerNodes int
	// The numbers of genes (links) in winner genome or zero if not solved
	WinnerGenes int
}

// Epochs is a sortable collection of epochs by execution time and Id
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
