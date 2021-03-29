package experiment

import (
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"time"
)

// The Experiment execution entry point
func (e *Experiment) Execute(context *neat.NeatContext, startGenome *genetics.Genome, executor interface{}) (err error) {
	if e.Trials == nil {
		e.Trials = make(Trials, context.NumRuns)
	}

	var pop *genetics.Population
	for run := 0; run < context.NumRuns; run++ {
		trialStartTime := time.Now()

		neat.InfoLog("\n>>>>> Spawning new population ")
		pop, err = genetics.NewPopulation(startGenome, context)
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
		epochExecutor, err := epochExecutorForContext(context)
		if err != nil {
			return err
		}

		// start new trial
		trial := Trial{
			Id: run,
		}

		if trialObserver, ok := executor.(TrialRunObserver); ok {
			trialObserver.TrialRunStarted(&trial) // optional
		}

		generationEvaluator := executor.(GenerationEvaluator) // mandatory

		for generationId := 0; generationId < context.NumGenerations; generationId++ {
			neat.InfoLog(fmt.Sprintf(">>>>> Generation:%3d\tRun: %d\n", generationId, run))
			generation := Generation{
				Id:      generationId,
				TrialId: run,
			}
			genStartTime := time.Now()
			err = generationEvaluator.GenerationEvaluate(pop, &generation, context)
			if err != nil {
				neat.InfoLog(fmt.Sprintf("!!!!! Generation [%d] evaluation failed !!!!!\n", generationId))
				return err
			}
			generation.Executed = time.Now()

			// Turnover population of organisms to the next epoch if appropriate
			if !generation.Solved {
				neat.DebugLog(">>>>> start next generation")
				err = epochExecutor.NextEpoch(generationId, pop, context)
				if err != nil {
					neat.InfoLog(fmt.Sprintf("!!!!! Epoch execution failed in generation [%d] !!!!!\n", generationId))
					return err
				}
			}

			// Set generation duration, which also includes preparation for the next epoch
			generation.Duration = generation.Executed.Sub(genStartTime)
			trial.Generations = append(trial.Generations, generation)

			if generation.Solved {
				// stop further evaluation if already solved
				neat.InfoLog(fmt.Sprintf(">>>>> The winner organism found in [%d] generation, fitness: %f <<<<<\n",
					generationId, generation.Best.Fitness))
				break
			}

		}
		// holds trial duration
		trial.Duration = time.Now().Sub(trialStartTime)

		// store trial into experiment
		e.Trials[run] = trial
	}

	return nil
}
