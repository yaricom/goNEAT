// Package neat implements NeuroEvolution of Augmenting Topologies (NEAT) method which can be used to evolve
// Artificial Neural Networks to perform specific task using genetic algorithms.
package neat

import (
	"math/rand"
	"fmt"
	"io"
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
	NumRuns                int

	// The flag to indicate whether to print additional debugging info
	IsDebugEnabled         bool

	/* OBSOLETE */
	recurProb float64
}

// Loads context configuration from provided reader
func LoadContext(r io.Reader) *NeatContext {
	c := NeatContext{}
	// read configuration
	var name string
	// trait_param_mut_prob 0.5
	fmt.Fscanf(r, "%s %f ", &name, &c.TraitParamMutProb)
	// trait_mutation_power 1.0
	fmt.Fscanf(r, "%s %f ", &name, &c.TraitMutationPower)
	// linktrait_mut_sig 1.0
	fmt.Fscanf(r, "%s %f ", &name, &c.LinkTraitMutSig)
	// nodetrait_mut_sig 0.5
	fmt.Fscanf(r, "%s %f ", &name, &c.NodeTraitMutSig)
	// weight_mut_power 1.8
	fmt.Fscanf(r, "%s %f ", &name, &c.WeightMutPower)
	// recur_prob 0.05
	fmt.Fscanf(r, "%s %f ", &name, &c.recurProb)
	// disjoint_coeff 1.0
	fmt.Fscanf(r, "%s %f ", &name, &c.DisjointCoeff)
	// excess_coeff 1.0
	fmt.Fscanf(r, "%s %f ", &name, &c.ExcessCoeff)
	// mutdiff_coeff 3.0
	fmt.Fscanf(r, "%s %f ", &name, &c.MutdiffCoeff)
	// compat_thresh 4.0
	fmt.Fscanf(r, "%s %f ", &name, &c.CompatThreshold)
	// age_significance 1.0
	fmt.Fscanf(r, "%s %f ", &name, &c.AgeSignificance)
	// survival_thresh 0.4
	fmt.Fscanf(r, "%s %f ", &name, &c.SurvivalThresh)
	// mutate_only_prob 0.25
	fmt.Fscanf(r, "%s %f ", &name, &c.MutateOnlyProb)
	// mutate_random_trait_prob 0.1
	fmt.Fscanf(r, "%s %f ", &name, &c.MutateRandomTraitProb)
	// mutate_link_trait_prob 0.1
	fmt.Fscanf(r, "%s %f ", &name, &c.MutateLinkTraitProb)
	// mutate_node_trait_prob 0.1
	fmt.Fscanf(r, "%s %f ", &name, &c.MutateNodeTraitProb)
	// mutate_link_weights_prob 0.8
	fmt.Fscanf(r, "%s %f ", &name, &c.MutateLinkWeightsProb)
	// mutate_toggle_enable_prob 0.1
	fmt.Fscanf(r, "%s %f ", &name, &c.MutateToggleEnableProb)
	// mutate_gene_reenable_prob 0.05
	fmt.Fscanf(r, "%s %f ", &name, &c.MutateGeneReenableProb)
	// mutate_add_node_prob 0.01
	fmt.Fscanf(r, "%s %f ", &name, &c.MutateAddNodeProb)
	// mutate_add_link_prob 0.3
	fmt.Fscanf(r, "%s %f ", &name, &c.MutateAddLinkProb)
	// interspecies_mate_rate 0.001
	fmt.Fscanf(r, "%s %f ", &name, &c.InterspeciesMateRate)
	// mate_multipoint_prob 0.6
	fmt.Fscanf(r, "%s %f ", &name, &c.MateMultipointProb)
	// mate_multipoint_avg_prob 0.4
	fmt.Fscanf(r, "%s %f ", &name, &c.MateMultipointAvgProb)
	// mate_singlepoint_prob 0.0
	fmt.Fscanf(r, "%s %f ", &name, &c.MateSinglepointProb)
	// mate_only_prob 0.2
	fmt.Fscanf(r, "%s %f ", &name, &c.MateOnlyProb)
	// recur_only_prob 0.2
	fmt.Fscanf(r, "%s %f ", &name, &c.RecurOnlyProb)
	// pop_size 1000
	fmt.Fscanf(r, "%s %d ", &name, &c.PopSize)
	// dropoff_age 15
	fmt.Fscanf(r, "%s %d ", &name, &c.DropOffAge)
	// newlink_tries 20
	fmt.Fscanf(r, "%s %d ", &name, &c.NewLinkTries)
	// print_every 60
	fmt.Fscanf(r, "%s %d ", &name, &c.PrintEvery)
	// babies_stolen 0
	fmt.Fscanf(r, "%s %d ", &name, &c.BabiesStolen)
	// num_runs 1
	fmt.Fscanf(r, "%s %d ", &name, &c.NumRuns)

	return &c
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
