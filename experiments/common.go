// The experiments package holds various experiments with NEAT.
package experiments

import (
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat"
	"fmt"
	"time"
)

// The type of action to be applied to environment
type ActionType byte

// The supported action types
const (
	// The continuous action type meaning continuous values to be applied to environment
	ContinuousAction ActionType = iota
	// The discrete action assumes that there are only discrete values of action (e.g. 0, 1)
	DiscreteAction
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


// The Experiment execution entry point
func (ex *Experiment) Execute(context *neat.NeatContext, start_genome *genetics.Genome, executor interface{}) (err error) {
	if ex.Trials == nil {
		ex.Trials = make(Trials, context.NumRuns)
	}

	var pop *genetics.Population
	for run := 0; run < context.NumRuns; run++ {
		neat.InfoLog("\n>>>>> Spawning new population ")
		pop, err = genetics.NewPopulation(start_genome, context)
		if err != nil {
			neat.InfoLog("Failed to spawn new population from start genome")
			return err
		} else {
			neat.InfoLog("OK <<<<<")
		}
		neat.InfoLog(">>>>> Verifying spawned population ")
		_, err = pop.Verify()
		if err != nil {
			neat.ErrorLog("\n!!!!! Population verification failed !!!!!")
			return err
		} else {
			neat.InfoLog("OK <<<<<")
		}

		// start new trial
		trial := Trial {
			Id:run,
		}

		if trial_observer, ok := executor.(TrialRunObserver); ok {
			trial_observer.TrialRunStarted(&trial) // optional
		}

		epoch_evaluator := executor.(GenerationEvaluator) // mandatory

		for generation_id := 0; generation_id < context.NumGenerations; generation_id++ {
			neat.InfoLog(fmt.Sprintf(">>>>> Generation:%3d\tRun: %d\n", generation_id, run))
			generation := Generation{
				Id:generation_id,
			}
			err = epoch_evaluator.GenerationEvaluate(pop, &generation, context)
			if err != nil {
				neat.InfoLog(fmt.Sprintf("!!!!! Generation [%d] evaluation failed !!!!!\n", generation_id))
				return err
			}
			generation.Executed = time.Now()
			trial.Epochs = append(trial.Epochs, generation)
			if generation.Solved {
				neat.InfoLog(fmt.Sprintf(">>>>> The winner organism found in [%d] generation! <<<<<\n", generation_id))
				break
			}
		}
		// store trial into experiment
		ex.Trials[run] = trial
	}

	return nil
}
