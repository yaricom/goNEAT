package pole

import (
	"context"
	"fmt"
	"github.com/yaricom/goNEAT/v2/experiment"
	"github.com/yaricom/goNEAT/v2/experiment/utils"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"math"
	"math/rand"
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

// GenerationEvaluate evaluates one epoch for given population and prints results into output directory if any.
func (e *cartPoleGenerationEvaluator) GenerationEvaluate(ctx context.Context, pop *genetics.Population, epoch *experiment.Generation) error {
	options, ok := neat.FromContext(ctx)
	if !ok {
		return neat.ErrNEATOptionsNotFound
	}
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
			epoch.WinnerEvals = options.PopSize*epoch.Id + org.Genotype.Id
			epoch.Best = org
			if epoch.WinnerNodes == 7 {
				// You could dump out optimal genomes here if desired
				if optPath, err := utils.WriteGenomePlain("pole1_optimal", e.OutputPath, org, epoch); err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump optimal genome, reason: %s\n", err))
				} else {
					neat.InfoLog(fmt.Sprintf("Dumped optimal genome to: %s\n", optPath))
				}
			}
		}
	}

	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(pop)

	// Only print to file every print_every generation
	if epoch.Solved || epoch.Id%options.PrintEvery == 0 {
		if _, err := utils.WritePopulationPlain(e.OutputPath, pop, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump population, reason: %s\n", err))
			return err
		}
	}

	if epoch.Solved {
		// print winner organism
		org := epoch.Best
		if depth, err := org.Phenotype.MaxActivationDepthFast(0); err == nil {
			neat.InfoLog(fmt.Sprintf("Activation depth of the winner: %d\n", depth))
		}

		genomeFile := "pole1_winner_genome"
		// Prints the winner organism to file!
		if orgPath, err := utils.WriteGenomePlain(genomeFile, e.OutputPath, org, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's genome, reason: %s\n", err))
		} else {
			neat.InfoLog(fmt.Sprintf("Generation #%d winner's genome dumped to: %s\n", epoch.Id, orgPath))
		}

		// Prints the winner organism's Phenotype to the Cytoscape JSON file!
		if orgPath, err := utils.WriteGenomeCytoscapeJSON(genomeFile, e.OutputPath, org, epoch); err != nil {
			neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism's phenome Cytoscape JSON graph, reason: %s\n", err))
		} else {
			neat.InfoLog(fmt.Sprintf("Generation #%d winner's phenome Cytoscape JSON graph dumped to: %s\n",
				epoch.Id, orgPath))
		}
	}

	return nil
}

// orgEvaluate evaluates provided organism for cart pole balancing task
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

	// Decide if it's a winner
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

// runCart runs the cart emulation and return number of emulation steps pole was balanced
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

	netDepth, err := net.MaxActivationDepthFast(0) // The max depth of the network to be activated
	if err != nil {
		neat.WarnLog(fmt.Sprintf(
			"Failed to estimate maximal depth of the network with loop.\nUsing default depth: %d", netDepth))
	} else if netDepth == 0 {
		// possibly disconnected - return minimal fitness score
		return 1, nil
	}

	in := make([]float64, 5)
	for steps = 0; steps < e.WinBalancingSteps; steps++ {
		/*-- set up the input layer based on the four inputs --*/
		in[0] = 1.0 // Bias
		in[1] = (x + 2.4) / 4.8
		in[2] = (xDot + .75) / 1.5
		in[3] = (theta + twelveDegrees) / .41
		in[4] = (thetaDot + 1.0) / 2.0
		if err = net.LoadSensors(in); err != nil {
			return 0, err
		}

		/*-- activate the network based on the input --*/
		if res, err := net.ForwardSteps(netDepth); !res {
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

// doAction was taken directly from the pole simulator written by Richard Sutton and Charles Anderson.
// This simulator uses normalized, continuous inputs instead of discretizing the input space.
/*----------------------------------------------------------------------
Takes an action (0 or 1) and the current values of the
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
