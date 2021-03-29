// The experiment package holds various experiments with NEAT.
package experiment

import (
	"errors"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"log"
	"os"
)

// The interface describing evaluator for one generation of evolution.
type GenerationEvaluator interface {
	// Invoked to evaluate one generation of population of organisms within given
	// execution context.
	GenerationEvaluate(pop *genetics.Population, epoch *Generation, context *neat.NeatContext) (err error)
}

// The interface to describe trial lifecycle observer interested to receive lifecycle notifications
type TrialRunObserver interface {
	// Invoked to notify that new trial run just started before any epoch evaluation in that trial run
	TrialRunStarted(trial *Trial)
}

// Returns appropriate executor type from given context
func epochExecutorForContext(context *neat.NeatContext) (genetics.PopulationEpochExecutor, error) {
	switch genetics.EpochExecutorType(context.EpochExecutorType) {
	case genetics.SequentialExecutorType:
		return &genetics.SequentialPopulationEpochExecutor{}, nil
	case genetics.ParallelExecutorType:
		return &genetics.ParallelPopulationEpochExecutor{}, nil
	default:
		return nil, errors.New("unsupported epoch executor type requested")
	}
}

// To provide standard output directory syntax based on current trial
// Method checks if directory should be created
func OutDirForTrial(outDir string, trialID int) string {
	dir := fmt.Sprintf("%s/%d", outDir, trialID)
	if _, err := os.Stat(dir); err != nil {
		// create output dir
		err := os.MkdirAll(dir, os.ModePerm)
		if err != nil {
			log.Fatal("Failed to create output directory: ", err)
		}
	}
	return dir
}
