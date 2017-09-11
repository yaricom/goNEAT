// Package neat implements NeuroEvolution of Augmenting Topologies (NEAT) method which can be used to evolve
// Artificial Neural Networks to perform specific task using genetic algorithms.
package neat

import "math/rand"

// The number of parameters used in neurons that learn through habituation,
// sensitization, or Hebbian-type processes
const Num_trait_params = 8

// The global configuration holder
type Neat struct {
	// Prob. of mutating a single trait param
	trait_param_mut_prob float64
	// Power of mutation on a single trait param
	trait_mutation_power float64
	// Amount that mutation_num changes for a trait change inside a link
	linktrait_mut_sig float64
	// Amount a mutation_num changes on a link connecting a node that changed its trait
	nodetrait_mut_sig float64
	// The power of a linkweight mutation
	weight_mut_power float64

	// These 3 global coefficients are used to determine the formula for
     	// computing the compatibility between 2 genomes.  The formula is:
     	// disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg.
     	// See the compatibility method in the Genome class for more info
     	// They can be thought of as the importance of disjoint Genes,
     	// excess Genes, and parametric difference between Genes of the
     	// same function, respectively.
	disjoint_coeff float64
	excess_coeff float64
	mutdiff_coeff float64

	// This global tells compatibility threshold under which
     	// two Genomes are considered the same species */
	compat_threshold float64

	/* Globals involved in the epoch cycle - mating, reproduction, etc.. */

	// How much does age matter?
	age_significance float64
	// Percent of ave fitness for survival
	survival_thresh float64

	// Probabilities of a non-mating reproduction
	mutate_only_prob float64
	mutate_random_trait_prob float64
	mutate_link_trait_prob float64
	mutate_node_trait_prob float64
	mutate_link_weights_prob float64
	mutate_toggle_enable_prob float64
	mutate_gene_reenable_prob float64
	mutate_add_node_prob float64
	mutate_add_link_prob float64

	// Probabilities of a mate being outside species
	interspecies_mate_rate float64
	mate_multipoint_prob float64
	mate_multipoint_avg_prob float64
	mate_singlepoint_prob float64

	// Prob. of mating without mutation
	mate_only_prob float64
	// Probability of forcing selection of ONLY links that are naturally recurrent
	recur_only_prob float64

	// Size of population
	pop_size int32
	// Age where Species starts to be penalized
	dropoff_age int32
	// Number of tries mutate_add_link will attempt to find an open link
	newlink_tries int32

	// Tells to print population to file every n generations
	print_every int32

	// The number of babies to siphen off to the champions
	babies_stolen int32

	// The number of runs to average over in an experiment
	num_runs int32
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
