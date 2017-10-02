// Package neat implements NeuroEvolution of Augmenting Topologies (NEAT) method which can be used to evolve
// Artificial Neural Networks to perform specific task using genetic algorithms.
package neat

import (
	"math/rand"
	"fmt"
)

// The number of parameters used in neurons that learn through habituation,
// sensitization, or Hebbian-type processes
const Num_trait_params = 8

// The NEAT execution context holding common configuration parameters, etc.
type NeatContext struct {
	// Prob. of mutating a single trait param
	TraitParamMutProb      float64
	// Power of mutation on a single trait param
	TraitMutationPower     float64
	// Amount that mutation_num changes for a trait change inside a link
	LinkTraitMutSig        float64
	// Amount a mutation_num changes on a link connecting a node that changed its trait
	NodeTraitMutSig        float64
	// The power of a linkweight mutation
	WeightMutPower         float64

	// These 3 global coefficients are used to determine the formula for
	// computing the compatibility between 2 genomes.  The formula is:
	// disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg.
	// See the compatibility method in the Genome class for more info
	// They can be thought of as the importance of disjoint Genes,
	// excess Genes, and parametric difference between Genes of the
	// same function, respectively.
	DisjointCoeff          float64
	ExcessCoeff            float64
	MutdiffCoeff           float64

	// This global tells compatibility threshold under which
	// two Genomes are considered the same species */
	CompatThreshold        float64

	/* Globals involved in the epoch cycle - mating, reproduction, etc.. */

	// How much does age matter?
	AgeSignificance        float64
	// Percent of ave fitness for survival
	SurvivalThresh         float64

	// Probabilities of a non-mating reproduction
	MutateOnlyProb         float64
	MutateRandomTraitProb  float64
	MutateLinkTraitProb    float64
	MutateNodeTraitProb    float64
	MutateLinkWeightsProb  float64
	MutateToggleEnableProb float64
	MutateGeneReenableProb float64
	MutateAddNodeProb      float64
	MutateAddLinkProb      float64

	// Probabilities of a mate being outside species
	InterspeciesMateRate   float64
	MateMultipointProb     float64
	MateMultipointAvgProb  float64
	MateSinglepointProb    float64

	// Prob. of mating without mutation
	MateOnlyProb           float64
	// Probability of forcing selection of ONLY links that are naturally recurrent
	RecurOnlyProb          float64

	// Size of population
	PopSize                int
	// Age when Species starts to be penalized
	DropOffAge             int
	// Number of tries mutate_add_link will attempt to find an open link
	NewLinkTries           int

	// Tells to print population to file every n generations
	PrintEvery             int

	// The number of babies to stolen off to the champions
	BabiesStolen           int

	// The number of runs to average over in an experiment
	Num_runs               int

	// The flag to indicate whether to print additional debugging info
	IsDebugEnabled         bool
}

func (c *NeatContext) DebugLog(rec string) {
	if c.IsDebugEnabled {
		fmt.Println(rec)
	}
}

// Returns
func RandPosNeg() int32 {
	v := rand.Int()
	if (v % 2) == 0 {
		return -1
	} else {
		return 1
	}
}
