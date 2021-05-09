package pole

import (
	"fmt"
	"github.com/yaricom/goNEAT/v2/experiment"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"math"
	"math/rand"
	"os"
)

const twelveDegrees = 12.0 * math.Pi / 180.0

type cartPoleGenerationEvaluator struct {
	// The output path to store execution results
	OutputPath string
	// The flag to indicate if cart emulator should be started from random position
	RandomStart bool
	// The number of emulation steps to be done balancing pole to win
	WinBalancingSteps int
}

// NewCartPoleGenerationEvaluator is to create generations evaluator for single-pole balancing experiment.
// This experiment performs evolution on single pole balancing task in order to produce appropriate genome.
func NewCartPoleGenerationEvaluator(outDir string, randomStart bool, winBalanceSteps int) experiment.GenerationEvaluator {
	return &cartPoleGenerationEvaluator{
		OutputPath:        outDir,
		RandomStart:       randomStart,
		WinBalancingSteps: winBalanceSteps,
	}
}

// GenerationEvaluate This method evaluates one epoch for given population and prints results into output directory if any.
func (e *cartPoleGenerationEvaluator) GenerationEvaluate(pop *genetics.Population, epoch *experiment.Generation, context *neat.Options) (err error) {
	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		res, err := e.orgEvaluate(org)
		if err != nil {
			return err
		}

		if res && (epoch.Best == nil || org.Fitness > epoch.Best.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize*epoch.Id + org.Genotype.Id
			epoch.Best = org
			if epoch.WinnerNodes == 7 {
				// You could dump out optimal genomes here if desired
				optPath := fmt.Sprintf("%s/%s_%d-%d", experiment.OutDirForTrial(e.OutputPath, epoch.TrialId),
					"pole1_optimal", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				if file, err := os.Create(optPath); err != nil {
					return err
				} else if err = org.Genotype.Write(file); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
					return err
				} else {
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", optPath))
				}
			}
		}
	}

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(pop)

	// Only print to file every print_every generations
	if epoch.Solved || epoch.Id%context.PrintEvery == 0 {
		popPath := fmt.Sprintf("%s/gen_%d", experiment.OutDirForTrial(e.OutputPath, epoch.TrialId), epoch.Id)
		if file, err := os.Create(popPath); err != nil {
			return err
		} else if err = pop.WriteBySpecies(file); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
			return err
		}
	}

	if epoch.Solved {
		// print winner organism
		for _, org := range pop.Organisms {
			if org.IsWinner {
				// Prints the winner organism to file!
				orgPath := fmt.Sprintf("%s/%s_%d-%d", experiment.OutDirForTrial(e.OutputPath, epoch.TrialId),
					"pole1_winner", org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				if file, err := os.Create(orgPath); err != nil {
					return err
				} else if err = org.Genotype.Write(file); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism genome, reason: %s\n", err))
					return err
				} else {
					neat.InfoLog(fmt.Sprintf("Generation #%d winner dumped to: %s\n", epoch.Id, orgPath))
				}
				break
			}
		}
	}

	return err
}

// This methods evaluates provided organism for cart pole balancing task
func (e *cartPoleGenerationEvaluator) orgEvaluate(organism *genetics.Organism) (bool, error) {
	// Try to balance a pole now
	if fitness, err := e.runCart(organism.Phenotype); err != nil {
		return false, nil
	} else {
		organism.Fitness = float64(fitness)
	}

	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf("Organism #%3d\tfitness: %f", organism.Genotype.Id, organism.Fitness))
	}

	// Decide if its a winner
	if organism.Fitness >= float64(e.WinBalancingSteps) {
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
		logSteps := math.Log(float64(e.WinBalancingSteps))
		organism.Error = (logSteps - math.Log(organism.Fitness)) / logSteps
		organism.Fitness = 1.0 - organism.Error
	}

	return organism.IsWinner, nil
}

// run cart emulation and return number of emulation steps pole was balanced
func (e *cartPoleGenerationEvaluator) runCart(net *network.Network) (steps int, err error) {
	var x float64        /* cart position, meters */
	var xDot float64     /* cart velocity */
	var theta float64    /* pole angle, radians */
	var thetaDot float64 /* pole angular velocity */
	if e.RandomStart {
		/*set up random start state*/
		x = float64(rand.Int31()%4800)/1000.0 - 2.4
		xDot = float64(rand.Int31()%2000)/1000.0 - 1
		theta = float64(rand.Int31()%400)/1000.0 - .2
		thetaDot = float64(rand.Int31()%3000)/1000.0 - 1.5
	}

	in := make([]float64, 5)
	for steps = 0; steps < e.WinBalancingSteps; steps++ {
		/*-- setup the input layer based on the four inputs --*/
		in[0] = 1.0 // Bias
		in[1] = (x + 2.4) / 4.8
		in[2] = (xDot + .75) / 1.5
		in[3] = (theta + twelveDegrees) / .41
		in[4] = (thetaDot + 1.0) / 2.0
		if err = net.LoadSensors(in); err != nil {
			return 0, err
		}

		/*-- activate the network based on the input --*/
		if res, err := net.Activate(); !res {
			//If it loops, exit returning only fitness of 1 step
			neat.DebugLog(fmt.Sprintf("Failed to activate Network, reason: %s", err))
			return 1, nil
		}
		/*-- decide which way to push via which output unit is greater --*/
		action := 1
		if net.Outputs[0].Activation > net.Outputs[1].Activation {
			action = 0
		}
		/*--- Apply action to the simulated cart-pole ---*/
		x, xDot, theta, thetaDot = e.doAction(action, x, xDot, theta, thetaDot)

		/*--- Check for failure.  If so, return steps ---*/
		if x < -2.4 || x > 2.4 || theta < -twelveDegrees || theta > twelveDegrees {
			return steps, nil
		}
	}
	return steps, nil
}

// cart_and_pole() was take directly from the pole simulator written by Richard Sutton and Charles Anderson.
// This simulator uses normalized, continuous inputs instead of discretizing the input space.
/*----------------------------------------------------------------------
cart_pole:  Takes an action (0 or 1) and the current values of the
four state variables and updates their values by estimating the state
TAU seconds later.
----------------------------------------------------------------------*/
func (e *cartPoleGenerationEvaluator) doAction(action int, x, xDot, theta, thetaDot float64) (xRet, xDotRet, thetaRet, thetaDotRet float64) {
	// The cart pole configuration values
	const Gravity = 9.8
	const MassCart = 1.0
	const MassPole = 0.5
	const TotalMass = MassPole + MassCart
	const Length = 0.5 /* actually half the pole's length */
	const PoleMassLength = MassPole * Length
	const ForceMag = 10.0
	const Tau = 0.02 /* seconds between state updates */
	const FourThirds = 1.3333333333333

	force := -ForceMag
	if action > 0 {
		force = ForceMag
	}
	cosTheta := math.Cos(theta)
	sinTheta := math.Sin(theta)

	temp := (force + PoleMassLength*thetaDot*thetaDot*sinTheta) / TotalMass

	thetaAcc := (Gravity*sinTheta - cosTheta*temp) / (Length * (FourThirds - MassPole*cosTheta*cosTheta/TotalMass))

	xAcc := temp - PoleMassLength*thetaAcc*cosTheta/TotalMass

	/*** Update the four state variables, using Euler's method. ***/
	xRet = x + Tau*xDot
	xDotRet = xDot + Tau*xAcc
	thetaRet = theta + Tau*thetaDot
	thetaDotRet = thetaDot + Tau*thetaAcc

	return xRet, xDotRet, thetaRet, thetaDotRet
}
