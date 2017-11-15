// The single pole balancing experiment is classic Reinforced Learning task proposed by Richard Sutton and Charles Anderson.
// In this experiment we will try to teach RF model of balancing pole placed on the moving cart.
package pole

import (
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/experiments"
	"fmt"
	"time"
)

// The pole balancing experiment entry point.
// This experiment performs evolution on single pole balancing task in order to produce appropriate genome.
func CartPoleExperiment(context *neat.NeatContext, start_genome *genetics.Genome, out_dir_path string, experiment *experiments.Experiment) (err error) {
	if experiment.Trials == nil {
		experiment.Trials = make(experiments.Trials, context.NumRuns)
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
		trial := experiments.Trial{
			Id:run,
		}

		for gen := 0; gen < context.NumGenerations; gen++ {
			neat.InfoLog(fmt.Sprintf(">>>>> Epoch: %d\tRun: %d\n", gen, run))
			epoch := experiments.Epoch{
				Id:gen,
			}
			err = pole_epoch(pop, gen, out_dir_path, &epoch, context)
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
		experiment.Trials[run] = trial
	}

	return nil
}

// Evaluate one pole balancing epoch
func pole_epoch(pop *genetics.Population, generation int, out_dir_path string, epoch *experiments.Epoch, context *neat.NeatContext) (err error) {

}
