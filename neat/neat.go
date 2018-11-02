// Package neat implements NeuroEvolution of Augmenting Topologies (NEAT) method which can be used to evolve
// Artificial Neural Networks to perform specific task using genetic algorithms.
package neat

import (
	"math/rand"
	"fmt"
	"io"
	"log"
	"os"
)

// LoggerLevel type to specify logger output level
type LoggerLevel byte

const (
	// The Debug log level
	LogLevelDebug LoggerLevel = iota
	// The Info log level
	LogLevelInfo
	// The Warning log level
	LogLevelWarning
	// The Error log level
	LogLevelError
)

var (
	// The current log level of the context
	LogLevel LoggerLevel

	loggerDebug = log.New(os.Stdout, "DEBUG: ", log.Ltime | log.Lshortfile)
	loggerInfo = log.New(os.Stdout, "INFO: ", log.Ltime | log.Lshortfile)
	loggerWarn = log.New(os.Stdout, "ALERT: ", log.Ltime | log.Lshortfile)
	loggerError = log.New(os.Stderr, "ERROR: ", log.Ltime | log.Lshortfile)

	// The logger to output all messages
	DebugLog = func(message string) {
		if LogLevel <= LogLevelDebug {
			loggerDebug.Output(2, message)
		}
	}
	// The logger to output messages with Info and up level
	InfoLog = func(message string) {
		if LogLevel <= LogLevelInfo {
			loggerInfo.Output(2, message)
		}
	}
	// The logger to output messages with Warn and up level
	WarnLog = func(message string) {
		if LogLevel <= LogLevelWarning {
			loggerWarn.Output(2, message)
		}
	}
	// The logger to output messages with Error and up level
	ErrorLog = func(message string) {
		if LogLevel <= LogLevelError {
			loggerError.Output(2, message)
		}
	}
)


// The NEAT execution context holding common configuration parameters, etc.
type NeatContext struct {
				       // Probability of mutating a single trait param
	TraitParamMutProb      float64
				       // Power of mutation on a single trait param
	TraitMutationPower     float64
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
	MutateConnectSensors   float64 // probability of mutation involving disconnected inputs connection

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

				       // The number of epochs (generations) to execute training
	NumGenerations         int
				       // The epoch's executor type to apply
	EpochExecutorType      int
}

// Loads context configuration from provided reader
func LoadContext(r io.Reader) *NeatContext {
	c := NeatContext{}
	// read configuration
	var name string
	var param float64;
	for true {
		_, err := fmt.Fscanf(r, "%s %f", &name, &param)
		if err == io.EOF {
			break
		}
		switch name {
		case "trait_param_mut_prob":
			c.TraitParamMutProb = param
		case "trait_mutation_power":
			c.TraitMutationPower = param
		case "weight_mut_power":
			c.WeightMutPower = param
		case "disjoint_coeff":
			c.DisjointCoeff = param
		case "excess_coeff":
			c.ExcessCoeff = param
		case "mutdiff_coeff":
			c.MutdiffCoeff = param
		case "compat_threshold":
			c.CompatThreshold = param
		case "age_significance":
			c.AgeSignificance = param
		case "survival_thresh":
			c.SurvivalThresh = param
		case "mutate_only_prob":
			c.MutateOnlyProb = param
		case "mutate_random_trait_prob":
			c.MutateRandomTraitProb = param
		case "mutate_link_trait_prob":
			c.MutateLinkTraitProb = param
		case "mutate_node_trait_prob":
			c.MutateNodeTraitProb = param
		case "mutate_link_weights_prob":
			c.MutateLinkWeightsProb = param
		case "mutate_toggle_enable_prob":
			c.MutateToggleEnableProb = param
		case "mutate_gene_reenable_prob":
			c.MutateGeneReenableProb = param
		case "mutate_add_node_prob":
			c.MutateAddNodeProb = param
		case "mutate_add_link_prob":
			c.MutateAddLinkProb = param
		case "mutate_connect_sensors":
			c.MutateConnectSensors = param
		case "interspecies_mate_rate":
			c.InterspeciesMateRate = param
		case "mate_multipoint_prob":
			c.MateMultipointProb = param
		case "mate_multipoint_avg_prob":
			c.MateMultipointAvgProb = param
		case "mate_singlepoint_prob":
			c.MateSinglepointProb = param
		case "mate_only_prob":
			c.MateOnlyProb = param
		case "recur_only_prob":
			c.RecurOnlyProb = param
		case "pop_size":
			c.PopSize = int(param)
		case "dropoff_age":
			c.DropOffAge = int(param)
		case "newlink_tries":
			c.NewLinkTries = int(param)
		case "print_every":
			c.PrintEvery = int(param)
		case "babies_stolen":
			c.BabiesStolen = int(param)
		case "num_runs":
			c.NumRuns = int(param)
		case "num_generations":
			c.NumGenerations = int(param)
		case "epoch_executor":
			c.EpochExecutorType = int(param)
		case "log_level":
			LogLevel = LoggerLevel(param)
		default:
			fmt.Printf("WARNING! Unknown configuration parameter found: %s = %f\n", name, param)
		}
	}

	return &c
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
