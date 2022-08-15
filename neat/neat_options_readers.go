package neat

import (
	"fmt"
	"github.com/pkg/errors"
	"github.com/spf13/cast"
	"github.com/yaricom/goNEAT/v3/neat/math"
	"gopkg.in/yaml.v3"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

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

// ReadNeatOptionsFromFile reads NEAT options from specified configFilePath automatically resolving config file encoding.
func ReadNeatOptionsFromFile(configFilePath string) (*Options, error) {
	configFile, err := os.Open(configFilePath)
	if err != nil {
		return nil, errors.Wrap(err, "failed to open config file")
	}
	fileName := configFile.Name()
	if strings.HasSuffix(fileName, "yml") || strings.HasSuffix(fileName, "yaml") {
		return LoadYAMLOptions(configFile)
	} else {
		return LoadNeatOptions(configFile)
	}
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
		if c.NodeActivatorsProb[i], err = strconv.ParseFloat(fields[1], 64); err != nil {
			return err
		}
	}
	return nil
}
