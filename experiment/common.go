// Package experiment defines standard evolutionary epochs evaluators and experimental data samples collectors.
package experiment

import (
	"errors"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
)

// GenerationEvaluator the interface describing evaluator for one epoch (generation) of the evolutionary process.
type GenerationEvaluator interface {
	// GenerationEvaluate Invoked to evaluate one generation of population of organisms within given
	// execution context.
	GenerationEvaluate(pop *genetics.Population, epoch *Generation, context *neat.Options) (err error)
}

// TrialRunObserver defines observer to be notified about experiment's trial lifecycle methods
type TrialRunObserver interface {
	// TrialRunStarted invoked to notify that new trial run just started. Invoked before any epoch evaluation in that trial run
	TrialRunStarted(trial *Trial)
	// TrialRunFinished invoked to notify that the trial run just finished. Invoked after all epochs evaluated or successful solver found.
	TrialRunFinished(trial *Trial)
	// EpochEvaluated invoked to notify that evaluation of specific epoch completed.
	EpochEvaluated(trial *Trial, epoch *Generation)
}

// Returns appropriate executor type from given context
func epochExecutorForContext(context *neat.Options) (genetics.PopulationEpochExecutor, error) {
	switch context.EpochExecutorType {
	case neat.EpochExecutorTypeSequential:
		return &genetics.SequentialPopulationEpochExecutor{}, nil
	case neat.EpochExecutorTypeParallel:
		return &genetics.ParallelPopulationEpochExecutor{}, nil
	default:
		return nil, errors.New("unsupported epoch executor type requested")
	}
}
