package experiment

import (
	"context"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"time"
)

// Execute is to run specific experiment using provided startGenome and specific evaluator for each epoch of the experiment
func (e *Experiment) Execute(ctx context.Context, startGenome *genetics.Genome, evaluator GenerationEvaluator, trialObserver TrialRunObserver) error {
	opts, found := neat.FromContext(ctx)
	if !found {
		return neat.ErrNEATOptionsNotFound
	}

	if e.Trials == nil {
		e.Trials = make(Trials, opts.NumRuns)
	}

	for run := 0; run < opts.NumRuns; run++ {
		trialStartTime := time.Now()

		neat.InfoLog("\n>>>>> Spawning new population ")
		pop, err := genetics.NewPopulation(startGenome, opts)
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
		epochExecutor, err := epochExecutorForContext(opts)
		if err != nil {
			return err
		}

		// start new trial
		trial := Trial{
			Id: run,
		}

		if trialObserver != nil {
			trialObserver.TrialRunStarted(&trial) // optional
		}

		for generationId := 0; generationId < opts.NumGenerations; generationId++ {
			// check if context was canceled
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			neat.InfoLog(fmt.Sprintf(">>>>> Generation:%3d\tRun: %d\n", generationId, run))
			generation := Generation{
				Id:      generationId,
				TrialId: run,
			}
			genStartTime := time.Now()
			err = evaluator.GenerationEvaluate(pop, &generation, opts)
			if err != nil {
				neat.InfoLog(fmt.Sprintf("!!!!! Generation [%d] evaluation failed !!!!!\n", generationId))
				return err
			}
			generation.Executed = time.Now()

			// Turnover population of organisms to the next epoch if appropriate
			if !generation.Solved {
				neat.DebugLog(">>>>> start next generation")
				err = epochExecutor.NextEpoch(ctx, generationId, pop)
				if err != nil {
					neat.InfoLog(fmt.Sprintf("!!!!! Epoch execution failed in generation [%d] !!!!!\n", generationId))
					return err
				}
			}

			// Set generation duration, which also includes preparation for the next epoch
			generation.Duration = generation.Executed.Sub(genStartTime)
			trial.Generations = append(trial.Generations, generation)

			// notify trial observer
			if trialObserver != nil {
				trialObserver.EpochEvaluated(&trial, &generation)
			}

			if generation.Solved {
				// stop further evaluation if already solved
				neat.InfoLog(fmt.Sprintf(">>>>> The winner organism found in [%d] generation, fitness: %f <<<<<\n",
					generationId, generation.Best.Fitness))
				break
			}
		}
		// holds trial duration
		trial.Duration = time.Since(trialStartTime)

		// store trial into experiment
		e.Trials[run] = trial

		// notify trial observer
		if trialObserver != nil {
			trialObserver.TrialRunFinished(&trial)
		}
	}

	return nil
}
