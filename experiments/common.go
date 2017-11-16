// The experiments package holds various experiments with NEAT.
package experiments

import (
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/neat"
	"fmt"
	"time"
)

// The interface describing evaluator for one epoch of evolution.
type EpochEvaluator interface {
	EpochEvaluate(pop *genetics.Population, epoch *Epoch, context *neat.NeatContext) (err error)
}


// The Experiment execution entry point
func (ex *Experiment) Execute(context *neat.NeatContext, start_genome *genetics.Genome, epoch_executor EpochEvaluator) (err error) {
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

		for gen := 0; gen < context.NumGenerations; gen++ {
			neat.InfoLog(fmt.Sprintf(">>>>> Epoch: %d\tRun: %d\n", gen, run))
			epoch := Epoch{
				Id:gen,
			}
			err = epoch_executor.EpochEvaluate(pop, &epoch, context)
			if err != nil {
				neat.InfoLog(fmt.Sprintf("!!!!! Epoch %d evaluation failed !!!!!\n", gen))
				return err
			}
			epoch.Executed = time.Now()
			trial.Epochs = append(trial.Epochs, epoch)
			if epoch.Solved {
				neat.InfoLog(fmt.Sprintf(">>>>> The winner organism found in epoch %d! <<<<<\n", gen))
				break
			}
		}
		// store trial into experiment
		ex.Trials[run] = trial
	}

	return nil
}
