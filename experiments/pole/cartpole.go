package pole

import (
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/experiments"
	"math"
	"github.com/yaricom/goNEAT/neat/network"
	"math/rand"
	"fmt"
	"os"
)

const twelve_degrees = 12.0 * math.Pi / 180.0

// The single pole balancing experiment entry point.
// This experiment performs evolution on single pole balancing task in order to produce appropriate genome.
type CartPoleGenerationEvaluator struct {
	// The output path to store execution results
	OutputPath        string
	// The flag to indicate if cart emulator should be started from random position
	RandomStart       bool
	// The number of emulation steps to be done balancing pole to win
	WinBalancingSteps int
}

// This method evaluates one epoch for given population and prints results into output directory if any.
func (ex CartPoleGenerationEvaluator) GenerationEvaluate(pop *genetics.Population, epoch *experiments.Generation, context *neat.NeatContext) (err error) {
	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		res := ex.orgEvaluate(org)

		if res && (epoch.Best == nil || org.Fitness > epoch.Best.Fitness){
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize * epoch.Id + org.Genotype.Id
			epoch.Best = org
			if (epoch.WinnerNodes == 7) {
				// You could dump out optimal genomes here if desired
				opt_path := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(ex.OutputPath, epoch.TrialId),
					"pole1_optimal", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				file, err := os.Create(opt_path)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
				} else {
					org.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", opt_path))
				}
			}
		}
	}

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(pop)

	// Only print to file every print_every generations
	if epoch.Solved || epoch.Id % context.PrintEvery == 0 {
		pop_path := fmt.Sprintf("%s/gen_%d", experiments.OutDirForTrial(ex.OutputPath, epoch.TrialId), epoch.Id)
		file, err := os.Create(pop_path)
		if err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
		} else {
			pop.WriteBySpecies(file)
		}
	}

	if epoch.Solved {
		// print winner organism
		for _, org := range pop.Organisms {
			if org.IsWinner {
				// Prints the winner organism to file!
				org_path := fmt.Sprintf("%s/%s_%d-%d", experiments.OutDirForTrial(ex.OutputPath, epoch.TrialId),
					"pole1_winner", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				file, err := os.Create(org_path)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism genome, reason: %s\n", err))
				} else {
					org.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Generation #%d winner dumped to: %s\n", epoch.Id, org_path))
				}
				break
			}
		}
	}

	return err
}

// This methods evaluates provided organism for cart pole balancing task
func (ex *CartPoleGenerationEvaluator) orgEvaluate(organism *genetics.Organism) bool {
	// Try to balance a pole now
	organism.Fitness = float64(ex.runCart(organism.Phenotype))

	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf("Organism #%3d\tfitness: %f", organism.Genotype.Id, organism.Fitness))
	}

	// Decide if its a winner
	if organism.Fitness >= float64(ex.WinBalancingSteps) {
		organism.IsWinner = true
	}

	// adjust fitness to be in range [0;1]
	if organism.IsWinner {
		organism.Fitness = 1.0
		organism.Error = 0.0
	} else if organism.Fitness == 0 {
		organism.Error = 1.0
	} else {
		// we use logarithmic scale because most cart runs fail to early within ~100 steps, but
		// we test against 500'000 balancing steps
		logSteps := math.Log(float64(ex.WinBalancingSteps))
		organism.Error = (logSteps - math.Log(organism.Fitness)) / logSteps
		organism.Fitness = 1.0 - organism.Error
	}

	return organism.IsWinner
}

// run cart emulation and return number of emulation steps pole was balanced
func (ex *CartPoleGenerationEvaluator) runCart(net *network.Network) (steps int) {
	var x float64           /* cart position, meters */
	var x_dot float64       /* cart velocity */
	var theta float64       /* pole angle, radians */
	var theta_dot float64   /* pole angular velocity */
	if ex.RandomStart {
		/*set up random start state*/
		x = float64(rand.Int31() % 4800) / 1000.0 - 2.4
		x_dot = float64(rand.Int31() % 2000) / 1000.0 - 1
		theta = float64(rand.Int31() % 400) / 1000.0 - .2
		theta_dot = float64(rand.Int31() % 3000) / 1000.0 - 1.5
	}

	in := make([]float64, 5)
	for steps = 0; steps < ex.WinBalancingSteps; steps++ {
		/*-- setup the input layer based on the four inputs --*/
		in[0] = 1.0  // Bias
		in[1] = (x + 2.4) / 4.8
		in[2] = (x_dot + .75) / 1.5
		in[3] = (theta + twelve_degrees) / .41
		in[4] = (theta_dot + 1.0) / 2.0
		net.LoadSensors(in)

		/*-- activate the network based on the input --*/
		if res, err := net.Activate(); !res {
			//If it loops, exit returning only fitness of 1 step
			neat.DebugLog(fmt.Sprintf("Failed to activate Network, reason: %s", err))
			return 1
		}
		/*-- decide which way to push via which output unit is greater --*/
		action := 1
		if net.Outputs[0].Activation > net.Outputs[1].Activation {
			action = 0
		}
		/*--- Apply action to the simulated cart-pole ---*/
		x, x_dot, theta, theta_dot = ex.doAction(action, x, x_dot, theta, theta_dot)

		/*--- Check for failure.  If so, return steps ---*/
		if (x < -2.4 || x > 2.4 || theta < -twelve_degrees || theta > twelve_degrees) {
			return steps
		}
	}
	return steps
}

// cart_and_pole() was take directly from the pole simulator written by Richard Sutton and Charles Anderson.
// This simulator uses normalized, continuous inputs instead of discretizing the input space.
/*----------------------------------------------------------------------
 cart_pole:  Takes an action (0 or 1) and the current values of the
 four state variables and updates their values by estimating the state
 TAU seconds later.
 ----------------------------------------------------------------------*/
func (ex *CartPoleGenerationEvaluator) doAction(action int, x, x_dot, theta, theta_dot float64) (x_ret, x_dot_ret, theta_ret, theta_dot_ret float64) {
	// The cart pole configuration values
	const GRAVITY = 9.8
	const MASSCART = 1.0
	const MASSPOLE = 0.5
	const TOTAL_MASS = (MASSPOLE + MASSCART)
	const LENGTH = 0.5      /* actually half the pole's length */
	const POLEMASS_LENGTH = (MASSPOLE * LENGTH)
	const FORCE_MAG = 10.0
	const TAU = 0.02      /* seconds between state updates */
	const FOURTHIRDS = 1.3333333333333

	force := -FORCE_MAG
	if action > 0 {
		force = FORCE_MAG
	}
	cos_theta := math.Cos(theta)
	sin_theta := math.Sin(theta)

	temp := (force + POLEMASS_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS

	theta_acc := (GRAVITY * sin_theta - cos_theta * temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * cos_theta * cos_theta / TOTAL_MASS))

	x_acc := temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

	/*** Update the four state variables, using Euler's method. ***/
	x_ret = x + TAU * x_dot
	x_dot_ret = x_dot + TAU * x_acc
	theta_ret = theta + TAU * theta_dot
	theta_dot_ret = theta_dot + TAU * theta_acc

	return x_ret, x_dot_ret, theta_ret, theta_dot_ret
}

