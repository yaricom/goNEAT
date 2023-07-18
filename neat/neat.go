// Package neat implements the NeuroEvolution of Augmenting Topologies (NEAT) method, which can be used to evolve
// specific Artificial Neural Networks from scratch using genetic algorithms.
package neat

import (
	"context"
	"fmt"
	"github.com/pkg/errors"
	"github.com/yaricom/goNEAT/v4/neat/math"
)

var (
	ErrNoActivatorsRegistered                = errors.New("no node activators registered with NEAT options, please assign at least one to NodeActivators")
	ErrActivatorsProbabilitiesNumberMismatch = errors.New("number of node activator probabilities doesn't match number of activators")
)

// GenomeCompatibilityMethod defines the method to calculate genomes compatibility
type GenomeCompatibilityMethod string

const (
	GenomeCompatibilityMethodLinear GenomeCompatibilityMethod = "linear"
	GenomeCompatibilityMethodFast   GenomeCompatibilityMethod = "fast"
)

// Validate is to check if this genome compatibility method supported by algorithm
func (g GenomeCompatibilityMethod) Validate() error {
	if g != GenomeCompatibilityMethodLinear && g != GenomeCompatibilityMethodFast {
		return errors.Errorf("unsupported genome compatibility method: [%s]", g)
	}
	return nil
}

// EpochExecutorType is to define the type of epoch evaluator
type EpochExecutorType string

const (
	EpochExecutorTypeSequential EpochExecutorType = "sequential"
	EpochExecutorTypeParallel   EpochExecutorType = "parallel"
)

// Validate is to check is this executor type is supported by algorithm
func (e EpochExecutorType) Validate() error {
	if e != EpochExecutorTypeSequential && e != EpochExecutorTypeParallel {
		return errors.Errorf("unsupported epoch executor type: [%s]", e)
	}
	return nil
}

// Options The NEAT algorithm options.
type Options struct {
	// Probability of mutating a single trait param
	TraitParamMutProb float64 `yaml:"trait_param_mut_prob"`
	// Power of mutation on a single trait param
	TraitMutationPower float64 `yaml:"trait_mutation_power"`
	// The power of a link weight mutation
	WeightMutPower float64 `yaml:"weight_mut_power"`

	// These 3 global coefficients are used to determine the formula for
	// computing the compatibility between 2 genomes.  The formula is:
	// disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg.
	// See the compatibility method in the Genome class for more info
	// They can be thought of as the importance of disjoint Genes,
	// excess Genes, and parametric difference between Genes of the
	// same function, respectively.
	DisjointCoeff float64 `yaml:"disjoint_coeff"`
	ExcessCoeff   float64 `yaml:"excess_coeff"`
	MutdiffCoeff  float64 `yaml:"mutdiff_coeff"`

	// This global tells compatibility threshold under which
	// two Genomes are considered the same species
	CompatThreshold float64 `yaml:"compat_threshold"`

	/* Globals involved in the epoch cycle - mating, reproduction, etc.. */

	// How much does age matter? Gives a fitness boost up to some young age (niching).
	// If it is 1, then young species get no fitness boost.
	AgeSignificance float64 `yaml:"age_significance"`
	// Percent of average fitness for survival, how many get to reproduce based on survival_thresh * pop_size
	SurvivalThresh float64 `yaml:"survival_thresh"`

	// Probabilities of a non-mating reproduction
	MutateOnlyProb         float64 `yaml:"mutate_only_prob"`
	MutateRandomTraitProb  float64 `yaml:"mutate_random_trait_prob"`
	MutateLinkTraitProb    float64 `yaml:"mutate_link_trait_prob"`
	MutateNodeTraitProb    float64 `yaml:"mutate_node_trait_prob"`
	MutateLinkWeightsProb  float64 `yaml:"mutate_link_weights_prob"`
	MutateToggleEnableProb float64 `yaml:"mutate_toggle_enable_prob"`
	MutateGeneReenableProb float64 `yaml:"mutate_gene_reenable_prob"`
	MutateAddNodeProb      float64 `yaml:"mutate_add_node_prob"`
	MutateAddLinkProb      float64 `yaml:"mutate_add_link_prob"`
	// probability of mutation involving disconnected inputs connection
	MutateConnectSensors float64 `yaml:"mutate_connect_sensors"`

	// Probabilities of a mate being outside species
	InterspeciesMateRate  float64 `yaml:"interspecies_mate_rate"`
	MateMultipointProb    float64 `yaml:"mate_multipoint_prob"`
	MateMultipointAvgProb float64 `yaml:"mate_multipoint_avg_prob"`
	MateSinglepointProb   float64 `yaml:"mate_singlepoint_prob"`

	// Prob. of mating without mutation
	MateOnlyProb float64 `yaml:"mate_only_prob"`
	// Probability of forcing selection of ONLY links that are naturally recurrent
	RecurOnlyProb float64 `yaml:"recur_only_prob"`

	// Size of population
	PopSize int `yaml:"pop_size"`
	// Age when Species starts to be penalized
	DropOffAge int `yaml:"dropoff_age"`
	// Number of tries mutate_add_link will attempt to find an open link
	NewLinkTries int `yaml:"newlink_tries"`

	// Tells to print population to file every n generations
	PrintEvery int `yaml:"print_every"`

	// The number of babies to stolen off to the champions
	BabiesStolen int `yaml:"babies_stolen"`

	// The number of runs to average over in an experiment
	NumRuns int `yaml:"num_runs"`

	// The number of epochs (generations) to execute training
	NumGenerations int `yaml:"num_generations"`

	// The epoch's executor type to apply (sequential, parallel)
	EpochExecutorType EpochExecutorType `yaml:"epoch_executor"`
	// The genome compatibility testing method to use (linear, fast (make sense for large genomes))
	GenCompatMethod GenomeCompatibilityMethod `yaml:"genome_compat_method"`

	// The neuron nodes activation functions list to choose from
	NodeActivators []math.NodeActivationType `yaml:"-"`
	// The probabilities of selection of the specific node activator function
	NodeActivatorsProb []float64 `yaml:"-"`

	// NodeActivatorsWithProbs the list of supported node activation with probability of each one
	NodeActivatorsWithProbs []string `yaml:"node_activators"`

	// LogLevel the log output details level
	LogLevel string `yaml:"log_level"`
}

// RandomNodeActivationType Returns next random node activation type among registered with this context
func (c *Options) RandomNodeActivationType() (math.NodeActivationType, error) {
	if len(c.NodeActivators) == 0 {
		return 0, ErrNoActivatorsRegistered
	}
	// quick check for the most cases
	if len(c.NodeActivators) == 1 {
		return c.NodeActivators[0], nil
	}

	// find random activator
	if len(c.NodeActivators) != len(c.NodeActivatorsProb) {
		return 0, ErrActivatorsProbabilitiesNumberMismatch
	}
	index := math.SingleRouletteThrow(c.NodeActivatorsProb)
	if index < 0 || index >= len(c.NodeActivators) {
		return 0, fmt.Errorf("unexpected error when trying to find random node activator, activator index: %d", index)
	}
	return c.NodeActivators[index], nil
}

// Validate is to validate that this options has valid values
func (c *Options) Validate() error {
	if err := c.EpochExecutorType.Validate(); err != nil {
		return err
	}

	if err := c.GenCompatMethod.Validate(); err != nil {
		return err
	}

	// check activators
	if len(c.NodeActivators) == 0 {
		return ErrNoActivatorsRegistered
	}
	if len(c.NodeActivators) != len(c.NodeActivatorsProb) {
		return ErrActivatorsProbabilitiesNumberMismatch
	}

	return nil
}

// NeatContext is to get Context which carries NEAT options inside to be propagated
func (c *Options) NeatContext() context.Context {
	return NewContext(context.Background(), c)
}
