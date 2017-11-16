package pole

import "github.com/yaricom/goNEAT/experiments"

// The double pole-balancing experiment both Markovian and non-Markovian versions
type CartDoublePoleEpochEvaluator struct {
	// The output path to store execution results
	OutputPath        string
	// The flag to indicate whether to apply Markovian evaluation variant
	Markovian bool
}

// The cart pole to hold state variables
type CartPole struct {

}

func (ev * CartDoublePoleEpochEvaluator) TrialRunStarted(trial *experiments.Trial) {

}


