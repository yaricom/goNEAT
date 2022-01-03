// Package neat implements the NeuroEvolution of Augmenting Topologies (NEAT) method, which can be used to evolve
// specific Artificial Neural Networks from scratch using genetic algorithms.
package neat

import (
	"context"
	"fmt"
	"github.com/pkg/errors"
	"github.com/spf13/cast"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"gopkg.in/yaml.v3"
	"io"
	"io/ioutil"
	"strconv"
	"strings"
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
	// quick check for the most cases
	if len(c.NodeActivators) == 1 {
		return c.NodeActivators[0], nil
	}
	// find next random
	index := math.SingleRouletteThrow(c.NodeActivatorsProb)
	if index < 0 || index >= len(c.NodeActivators) {
		return 0, fmt.Errorf("unexpected error when trying to find random node activator, activator index: %d", index)
	}
	return c.NodeActivators[index], nil
}

// set default values for activator type and its probability of selection
func (c *Options) initNodeActivators() (err error) {
	if len(c.NodeActivatorsWithProbs) == 0 {
		c.NodeActivators = []math.NodeActivationType{math.SigmoidSteepenedActivation}
		c.NodeActivatorsProb = []float64{1.0}
		return nil
	}
	// create activators
	actFns := c.NodeActivatorsWithProbs
	c.NodeActivators = make([]math.NodeActivationType, len(actFns))
	c.NodeActivatorsProb = make([]float64, len(actFns))
	for i, line := range actFns {
		fields := strings.Fields(line)
		if c.NodeActivators[i], err = math.NodeActivators.ActivationTypeFromName(fields[0]); err != nil {
			return err
		}
		if prob, err := strconv.ParseFloat(fields[1], 64); err != nil {
			return err
		} else {
			c.NodeActivatorsProb[i] = prob
		}
	}
	return nil
}

// Validate is to validate that this options has valid values
func (c *Options) Validate() error {
	if err := c.EpochExecutorType.Validate(); err != nil {
		return err
	}

	if err := c.GenCompatMethod.Validate(); err != nil {
		return err
	}
	return nil
}

// NeatContext is to get Context which carries NEAT options inside to be propagated
func (c *Options) NeatContext() context.Context {
	return NewContext(context.Background(), c)
}

// LoadYAMLOptions is to load NEAT options encoded as YAML file
func LoadYAMLOptions(r io.Reader) (*Options, error) {
	content, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	// read options
	var opts Options
	if err = yaml.Unmarshal(content, &opts); err != nil {
		return nil, errors.Wrap(err, "failed to decode NEAT options from YAML")
	}

	// initialize logger
	if err = InitLogger(opts.LogLevel); err != nil {
		return nil, errors.Wrap(err, "failed to initialize logger")
	}

	// read node activators
	if err = opts.initNodeActivators(); err != nil {
		return nil, errors.Wrap(err, "failed to read node activators")
	}

	if err = opts.Validate(); err != nil {
		return nil, errors.Wrap(err, "invalid NEAT options")
	}

	return &opts, nil
}

// LoadNeatOptions Loads NEAT options configuration from provided reader encode in plain text format (.neat)
func LoadNeatOptions(r io.Reader) (*Options, error) {
	c := &Options{}
	// read configuration
	var name string
	var param string
	for {
		_, err := fmt.Fscanf(r, "%s %v\n", &name, &param)
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		switch name {
		case "trait_param_mut_prob":
			c.TraitParamMutProb = cast.ToFloat64(param)
		case "trait_mutation_power":
			c.TraitMutationPower = cast.ToFloat64(param)
		case "weight_mut_power":
			c.WeightMutPower = cast.ToFloat64(param)
		case "disjoint_coeff":
			c.DisjointCoeff = cast.ToFloat64(param)
		case "excess_coeff":
			c.ExcessCoeff = cast.ToFloat64(param)
		case "mutdiff_coeff":
			c.MutdiffCoeff = cast.ToFloat64(param)
		case "compat_threshold":
			c.CompatThreshold = cast.ToFloat64(param)
		case "age_significance":
			c.AgeSignificance = cast.ToFloat64(param)
		case "survival_thresh":
			c.SurvivalThresh = cast.ToFloat64(param)
		case "mutate_only_prob":
			c.MutateOnlyProb = cast.ToFloat64(param)
		case "mutate_random_trait_prob":
			c.MutateRandomTraitProb = cast.ToFloat64(param)
		case "mutate_link_trait_prob":
			c.MutateLinkTraitProb = cast.ToFloat64(param)
		case "mutate_node_trait_prob":
			c.MutateNodeTraitProb = cast.ToFloat64(param)
		case "mutate_link_weights_prob":
			c.MutateLinkWeightsProb = cast.ToFloat64(param)
		case "mutate_toggle_enable_prob":
			c.MutateToggleEnableProb = cast.ToFloat64(param)
		case "mutate_gene_reenable_prob":
			c.MutateGeneReenableProb = cast.ToFloat64(param)
		case "mutate_add_node_prob":
			c.MutateAddNodeProb = cast.ToFloat64(param)
		case "mutate_add_link_prob":
			c.MutateAddLinkProb = cast.ToFloat64(param)
		case "mutate_connect_sensors":
			c.MutateConnectSensors = cast.ToFloat64(param)
		case "interspecies_mate_rate":
			c.InterspeciesMateRate = cast.ToFloat64(param)
		case "mate_multipoint_prob":
			c.MateMultipointProb = cast.ToFloat64(param)
		case "mate_multipoint_avg_prob":
			c.MateMultipointAvgProb = cast.ToFloat64(param)
		case "mate_singlepoint_prob":
			c.MateSinglepointProb = cast.ToFloat64(param)
		case "mate_only_prob":
			c.MateOnlyProb = cast.ToFloat64(param)
		case "recur_only_prob":
			c.RecurOnlyProb = cast.ToFloat64(param)
		case "pop_size":
			c.PopSize = cast.ToInt(param)
		case "dropoff_age":
			c.DropOffAge = cast.ToInt(param)
		case "newlink_tries":
			c.NewLinkTries = cast.ToInt(param)
		case "print_every":
			c.PrintEvery = cast.ToInt(param)
		case "babies_stolen":
			c.BabiesStolen = cast.ToInt(param)
		case "num_runs":
			c.NumRuns = cast.ToInt(param)
		case "num_generations":
			c.NumGenerations = cast.ToInt(param)
		case "epoch_executor":
			c.EpochExecutorType = EpochExecutorType(param)
		case "genome_compat_method":
			c.GenCompatMethod = GenomeCompatibilityMethod(param)
		case "log_level":
			c.LogLevel = param
		default:
			return nil, errors.Errorf("unknown configuration parameter found: %s = %s", name, param)
		}
	}
	// initialize logger
	if err := InitLogger(c.LogLevel); err != nil {
		return nil, errors.Wrap(err, "failed to initialize logger")
	}

	if err := c.initNodeActivators(); err != nil {
		return nil, err
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}

	return c, nil
}
