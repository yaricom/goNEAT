// The experiments package holds various experiments with NEAT.
package experiments

import (
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat"
	"fmt"
	"time"
	"os"
	"log"
	"errors"
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

// Returns appropriate executor type from given context
func epochExecutorForContext(context *neat.NeatContext) (genetics.PopulationEpochExecutor, error) {
	switch genetics.EpochExecutorType(context.EpochExecutorType) {
	case genetics.SequentialExecutorType:
		return &genetics.SequentialPopulationEpochExecutor{}, nil
	case genetics.ParallelExecutorType:
		return &genetics.ParallelPopulationEpochExecutor{}, nil
	default:
		return nil, errors.New("Unsupported epoch executor type requested")
	}
}

// The Experiment execution entry point
func (ex *Experiment) Execute(context *neat.NeatContext, start_genome *genetics.Genome, executor interface{}) (err error) {
	if ex.Trials == nil {
		ex.Trials = make(Trials, context.NumRuns)
	}

	var pop *genetics.Population
	for run := 0; run < context.NumRuns; run++ {
		trial_start_time := time.Now()

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

		// create appropriate population's epoch executor
		epoch_executor, err := epochExecutorForContext(context)
		if err != nil {
			return err
		}

		// start new trial
		trial := Trial {
			Id:run,
		}

		if trial_observer, ok := executor.(TrialRunObserver); ok {
			trial_observer.TrialRunStarted(&trial) // optional
		}

		generation_evaluator := executor.(GenerationEvaluator) // mandatory

		for generation_id := 0; generation_id < context.NumGenerations; generation_id++ {
			neat.InfoLog(fmt.Sprintf(">>>>> Generation:%3d\tRun: %d\n", generation_id, run))
			generation := Generation{
				Id:generation_id,
				TrialId:run,
			}
			gen_start_time := time.Now()
			err = generation_evaluator.GenerationEvaluate(pop, &generation, context)
			if err != nil {
				neat.InfoLog(fmt.Sprintf("!!!!! Generation [%d] evaluation failed !!!!!\n", generation_id))
				return err
			}
			generation.Executed = time.Now()

			// Turnover population of organisms to the next epoch if appropriate
			if !generation.Solved {
				neat.DebugLog(">>>>> start next generation")
				err = epoch_executor.NextEpoch(generation_id, pop, context)
				if err != nil {
					neat.InfoLog(fmt.Sprintf("!!!!! Epoch execution failed in generation [%d] !!!!!\n", generation_id))
					return err
				}
			}

			// Set generation duration, which also includes preparation for the next epoch
			generation.Duration = generation.Executed.Sub(gen_start_time)
			trial.Generations = append(trial.Generations, generation)

			if generation.Solved {
				// stop further evaluation if already solved
				neat.InfoLog(fmt.Sprintf(">>>>> The winner organism found in [%d] generation, fitness: %f <<<<<\n",
					generation_id, generation.Best.Fitness))
				break
			}

		}
		// holds trial duration
		trial.Duration = time.Now().Sub(trial_start_time)

		// store trial into experiment
		ex.Trials[run] = trial
	}

	return nil
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
