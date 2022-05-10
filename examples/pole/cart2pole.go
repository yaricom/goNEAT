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
	"sort"
)

const thirtySixDegrees = 36 * math.Pi / 180.0

// The maximal number of time steps for Markov experiment
const markovMaxSteps = 100000

// The maximal number of time steps for Non-Markov long run
const nonMarkovLongMaxSteps = 100000

// The maximal number of time steps for Non-Markov generalization run
const nonMarkovGeneralizationMaxSteps = 1000

type cartDoublePoleGenerationEvaluator struct {
	// The output path to store execution results
	OutputPath string
	// The flag to indicate whether to apply Markov evaluation variant
	Markov bool

	// The flag to indicate whether to use continuous activation or discrete
	ActionType ActionType
}

// NewCartDoublePoleGenerationEvaluator is the generations evaluator for double-pole balancing experiment: both Markov and non-Markov versions
func NewCartDoublePoleGenerationEvaluator(outDir string, markov bool, actionType ActionType) experiment.GenerationEvaluator {
	return &cartDoublePoleGenerationEvaluator{
		OutputPath: outDir,
		Markov:     markov,
		ActionType: actionType,
	}
}

// CartPole The structure to describe cart pole emulation
type CartPole struct {
	// The flag to indicate that we are executing Markov experiment setup (known velocities information)
	isMarkov bool
	// Flag that we are looking at the champion in Non-Markov experiment
	nonMarkovLong bool
	// Flag that we are testing champion's generalization
	generalizationTest bool

	// The state of the system (x, ∆x/∆t, θ1, ∆θ1/∆t, θ2, ∆θ2/∆t)
	state [6]float64

	// The number of balanced time steps passed for current organism evaluation
	balancedTimeSteps int

	jiggleStep [1000]float64

	// Queues used for Gruau's fitness which damps oscillations
	cartPosSum      float64
	cartVelocitySum float64
	polePosSum      float64
	poleVelocitySum float64
}

// GenerationEvaluate Perform evaluation of one epoch on double pole balancing
func (e *cartDoublePoleGenerationEvaluator) GenerationEvaluate(ctx context.Context, pop *genetics.Population, epoch *experiment.Generation) error {
	options, ok := neat.FromContext(ctx)
	if !ok {
		return neat.ErrNEATOptionsNotFound
	}
	cartPole := newCartPole(e.Markov)

	cartPole.nonMarkovLong = false
	cartPole.generalizationTest = false

	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		winner, err := e.orgEvaluate(org, cartPole)
		if err != nil {
			return err
		}

		if winner && (epoch.Best == nil || org.Fitness > epoch.Best.Fitness) {
			// This will be winner in Markov case
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = options.PopSize*epoch.Id + org.Genotype.Id
			epoch.Best = org
			org.IsWinner = true
		}
	}

	// Check for winner in Non-Markov case
	if !e.Markov {
		// The best individual (i.e. the one with the highest fitness value) of every generation is tested for
		// its ability to balance the system for a longer time period. If a potential solution passes this test
		// by keeping the system balanced for 100’000 time steps, the so called generalization score(GS) of this
		// particular individual is calculated. This score measures the potential of a controller to balance the
		// system starting from different initial conditions. It's calculated with a series of experiments, running
		// over 1000 time steps, starting from 625 different initial conditions.
		// The initial conditions are chosen by assigning each value of the set Ω = [0.05 0.25 0.5 0.75 0.95] to
		// each of the states x, ∆x/∆t, θ1 and ∆θ1/∆t, scaled to the range of the variables.The short pole angle θ2
		// and its angular velocity ∆θ2/∆t are set to zero. The GS is then defined as the number of successful runs
		// from the 625 initial conditions and an individual is defined as a solution if it reaches a generalization
		// score of 200 or more.

		// Sort the species by max organism fitness in descending order - the highest fitness first
		sortedSpecies := make([]*genetics.Species, len(pop.Species))
		copy(sortedSpecies, pop.Species)
		sort.Sort(sort.Reverse(genetics.ByOrganismFitness(sortedSpecies)))

		// First update what is checked and unchecked
		var currSpecies *genetics.Species
		for _, currSpecies = range sortedSpecies {
			max, _ := currSpecies.ComputeMaxAndAvgFitness()
			if max > currSpecies.MaxFitnessEver {
				currSpecies.IsChecked = false
			}
		}

		// Now find first (most fit) species that is unchecked
		currSpecies = nil
		for _, currSpecies = range sortedSpecies {
			if !currSpecies.IsChecked {
				break
			}
		}
		if currSpecies == nil {
			currSpecies = sortedSpecies[0]
		}

		// Remember it was checked
		currSpecies.IsChecked = true

		// the organism champion
		champion := currSpecies.FindChampion()
		championFitness := champion.Fitness

		// Now check to make sure the champion can do 100'000 evaluations
		cartPole.nonMarkovLong = true
		cartPole.generalizationTest = false

		longRunPassed, err := e.orgEvaluate(champion, cartPole)
		if err != nil {
			return err
		}
		if longRunPassed {

			// the champion passed non-Markov long test, start generalization
			cartPole.nonMarkovLong = false
			cartPole.generalizationTest = true

			// Given that the champion passed long run test, now run it on generalization tests running
			// over 1'000 time steps, starting from 625 different initial conditions
			stateVals := [5]float64{0.05, 0.25, 0.5, 0.75, 0.95}
			generalizationScore := 0
			for s0c := 0; s0c < 5; s0c++ {
				for s1c := 0; s1c < 5; s1c++ {
					for s2c := 0; s2c < 5; s2c++ {
						for s3c := 0; s3c < 5; s3c++ {
							cartPole.state[0] = stateVals[s0c]*4.32 - 2.16
							cartPole.state[1] = stateVals[s1c]*2.70 - 1.35
							cartPole.state[2] = stateVals[s2c]*0.12566304 - 0.06283152 // 0.06283152 = 3.6 degrees
							cartPole.state[3] = stateVals[s3c]*0.30019504 - 0.15009752 // 0.15009752 = 8.6 degrees
							// The short pole angle and its angular velocity are set to zero.
							cartPole.state[4] = 0.0
							cartPole.state[5] = 0.0

							// The champion needs to be flushed here because it may have
							// leftover activation from its last test run that could affect
							// its recurrent memory
							if _, err = champion.Phenotype.Flush(); err != nil {
								return err
							}

							if generalized, err := e.orgEvaluate(champion, cartPole); generalized {
								generalizationScore++

								if neat.LogLevel == neat.LogLevelDebug {
									neat.DebugLog(
										fmt.Sprintf("x: %f, xv: %f, t1: %f, t2: %f, angle: %f\n",
											cartPole.state[0], cartPole.state[1],
											cartPole.state[2], cartPole.state[4], thirtySixDegrees))
								}
							} else if err != nil {
								return err
							}
						}
					}
				}
			}

			if generalizationScore >= 200 {
				// The generalization test winner
				neat.InfoLog(
					fmt.Sprintf("The non-Markov champion found! (Generalization Score = %d)",
						generalizationScore))
				champion.Fitness = float64(generalizationScore)
				champion.IsWinner = true
				epoch.Solved = true
				epoch.WinnerNodes = len(champion.Genotype.Nodes)
				epoch.WinnerGenes = champion.Genotype.Extrons()
				epoch.WinnerEvals = options.PopSize*epoch.Id + champion.Genotype.Id
				epoch.Best = champion
			} else {
				neat.InfoLog("The non-Markov champion unable to generalize")
				champion.Fitness = championFitness // Restore the champ's fitness
				champion.IsWinner = false
			}
		} else {
			neat.InfoLog("The non-Markov champion missed the 100'000 run test")
			champion.Fitness = championFitness // Restore the champ's fitness
			champion.IsWinner = false
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
		// The max depth of the network to be activated
		if depth, err := org.Phenotype.MaxActivationDepthFast(0); err == nil {
			neat.InfoLog(fmt.Sprintf("Activation depth of the winner: %d\n", depth))
		}

		genomeFile := "pole2_winner_genome"
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

// orgEvaluate method evaluates fitness of the organism for cart double pole-balancing task
func (e *cartDoublePoleGenerationEvaluator) orgEvaluate(organism *genetics.Organism, cartPole *CartPole) (winner bool, err error) {
	// Try to balance a pole now
	organism.Fitness, err = cartPole.evalNet(organism.Phenotype, e.ActionType)
	if err != nil {
		return false, err
	}

	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf("Organism #%3d\tfitness: %f", organism.Genotype.Id, organism.Fitness))
	}

	// DEBUG CHECK if organism is damaged
	if !(cartPole.nonMarkovLong && cartPole.generalizationTest) && organism.CheckChampionChildDamaged() {
		neat.WarnLog(fmt.Sprintf("ORGANISM DEGRADED:\n%s", organism.Genotype))
	}

	// Decide if it's a winner, in Markov Case
	if cartPole.isMarkov {
		if organism.Fitness >= markovMaxSteps {
			winner = true
			organism.Fitness = 1.0
			organism.Error = 0.0
		} else {
			// we use linear scale
			organism.Error = (markovMaxSteps - organism.Fitness) / markovMaxSteps
			organism.Fitness = 1.0 - organism.Error
		}
	} else if cartPole.nonMarkovLong {
		// if doing the long test non-markov
		if organism.Fitness >= nonMarkovLongMaxSteps {
			winner = true
		}
	} else if cartPole.generalizationTest {
		if organism.Fitness >= nonMarkovGeneralizationMaxSteps {
			winner = true
		}
	} else {
		winner = false
	}
	return winner, err
}

// If markov is false, then velocity information will be withheld from the network population (non-Markov)
func newCartPole(markov bool) *CartPole {
	return &CartPole{
		isMarkov: markov,
	}
}

func (p *CartPole) evalNet(net *network.Network, actionType ActionType) (steps float64, err error) {
	nonMarkovMax := nonMarkovGeneralizationMaxSteps
	if p.nonMarkovLong {
		nonMarkovMax = nonMarkovLongMaxSteps
	}

	p.resetState()

	netDepth, err := net.MaxActivationDepthFast(0) // The max depth of the network to be activated
	if err != nil {
		neat.WarnLog(fmt.Sprintf(
			"Failed to estimate activation depth of the network, skipping evaluation: %s", err))
		return 0, nil
	} else if netDepth == 0 {
		// disconnected - assign minimal fitness to not completely exclude organism from evolution
		// returning only fitness of 1 step
		if p.isMarkov {
			return 1.0, nil
		} else {
			return 0.0001, err
		}
	}

	if p.isMarkov {
		input := make([]float64, 7)
		for steps = 0; steps < markovMaxSteps; steps++ {
			input[0] = (p.state[0] + 2.4) / 4.8
			input[1] = (p.state[1] + 1.0) / 2.0
			input[2] = (p.state[2] + thirtySixDegrees) / (thirtySixDegrees * 2.0) //0.52
			input[3] = (p.state[3] + 1.0) / 2.0
			input[4] = (p.state[4] + thirtySixDegrees) / (thirtySixDegrees * 2.0) //0.52
			input[5] = (p.state[5] + 1.0) / 2.0
			input[6] = 0.5

			if err = net.LoadSensors(input); err != nil {
				return 0, err
			}

			/*-- activate the network based on the input --*/
			if res, err := net.ActivateSteps(netDepth); err != nil {
				neat.DebugLog(fmt.Sprintf("Failed to activate Network, reason: %s", err))
				return 0, err
			} else if !res {
				// If it loops, exit returning only fitness of 1 step
				return 1.0, nil
			}
			action := net.Outputs[0].Activation
			if actionType == DiscreteAction {
				// make action values discrete
				if action < 0.5 {
					action = 0
				} else {
					action = 1
				}
			}
			p.performAction(action, steps)

			if p.outsideBounds() {
				// if failure stop it now
				break
			}

			//fmt.Printf("x: % f, xv: % f, t1: % f, t2: % f, angle: % f\n", p.state[0], p.state[1], p.state[2], p.state[4], thirty_six_degrees)
		}
		return steps, nil
	} else {
		input := make([]float64, 4)
		// The non Markov case
		for steps = 0; steps < float64(nonMarkovMax); steps++ {
			input[0] = p.state[0] / 4.8
			input[1] = p.state[2] / 0.52
			input[2] = p.state[4] / 0.52
			input[3] = 1.0

			err = net.LoadSensors(input)
			if err != nil {
				return 0, err
			}

			/*-- activate the network based on the input --*/
			if res, err := net.ActivateSteps(netDepth); err != nil {
				neat.WarnLog(fmt.Sprintf("Failed to activate Network, reason: %s", err))
				return 0, err
			} else if !res {
				// If it loops, exit returning only fitness of 1 step
				return 0.0001, nil
			}

			action := net.Outputs[0].Activation
			if actionType == DiscreteAction {
				// make action values discrete
				if action < 0.5 {
					action = 0
				} else {
					action = 1
				}
			}
			p.performAction(action, steps)
			if p.outsideBounds() {
				//fmt.Printf("x: % f, xv: % f, t1: % f, t2: % f, angle: % f, steps: %f\n",
				//	p.state[0], p.state[1], p.state[2], p.state[4], thirty_six_degrees, steps)
				// if failure stop it now
				break
			}

		}
		/*-- If we are generalizing we just need to balance it for a while --*/
		if p.generalizationTest {
			return float64(p.balancedTimeSteps), nil
		}

		// Sum last 100
		jiggleTotal := 0.0
		if steps >= 100.0 && !p.nonMarkovLong {
			// Adjust for array bounds and count
			for count := int(steps - 100.0); count < int(steps); count++ {
				jiggleTotal += p.jiggleStep[count]
			}
		}
		if !p.nonMarkovLong {
			var nonMarkovFitness float64
			if p.balancedTimeSteps >= 100 {
				// F = 0.1f1 + 0.9f2
				nonMarkovFitness = 0.1*float64(p.balancedTimeSteps)/1000.0 + 0.9*0.75/jiggleTotal
			} else {
				// F = t / 1000
				nonMarkovFitness = 0.1 * float64(p.balancedTimeSteps) / 1000.0
			}
			if neat.LogLevel == neat.LogLevelDebug {
				neat.DebugLog(fmt.Sprintf("Balanced time steps: %d, jiggle: %f ***\n",
					p.balancedTimeSteps, jiggleTotal))
			}
			return nonMarkovFitness, nil
		} else {
			return steps, nil
		}
	}
}

func (p *CartPole) performAction(action, stepNum float64) {
	const TAU = 0.01 // ∆t = 0.01s

	/*--- Apply action to the simulated cart-pole ---*/
	// Runge-Kutta 4th order integration method
	var dydx [6]float64
	for i := 0; i < 2; i++ {
		dydx[0] = p.state[1]
		dydx[2] = p.state[3]
		dydx[4] = p.state[5]
		p.step(action, p.state, &dydx)
		p.rk4(action, p.state, dydx, &p.state, TAU)
	}

	// Record this state
	p.cartPosSum += math.Abs(p.state[0])
	p.cartVelocitySum += math.Abs(p.state[1])
	p.polePosSum += math.Abs(p.state[2])
	p.poleVelocitySum += math.Abs(p.state[3])

	if stepNum < 1000 {
		p.jiggleStep[int(stepNum)] = math.Abs(p.state[0]) + math.Abs(p.state[1]) + math.Abs(p.state[2]) + math.Abs(p.state[3])
	}
	if !p.outsideBounds() {
		p.balancedTimeSteps++
	}
}

func (p *CartPole) step(action float64, st [6]float64, derivs *[6]float64) {
	const Mup = 0.000002
	const Gravity = -9.8
	const ForceMag = 10.0 // [N]
	const MassCart = 1.0  // [kg]

	const MassPole1 = 1.0 // [kg]
	const Length1 = 0.5   // [m] - actually half the first pole's length

	const Length2 = 0.05  // [m] - actually half the second pole's length
	const MassPole2 = 0.1 // [kg]

	force := (action - 0.5) * ForceMag * 2.0
	cosTheta1 := math.Cos(st[2])
	sinTheta1 := math.Sin(st[2])
	gSinTheta1 := Gravity * sinTheta1
	cosTheta2 := math.Cos(st[4])
	sinTheta2 := math.Sin(st[4])
	gSinTheta2 := Gravity * sinTheta2

	ml1 := Length1 * MassPole1
	ml2 := Length2 * MassPole2
	temp1 := Mup * st[3] / ml1
	temp2 := Mup * st[5] / ml2
	fi1 := (ml1 * st[3] * st[3] * sinTheta1) + (0.75 * MassPole1 * cosTheta1 * (temp1 + gSinTheta1))
	fi2 := (ml2 * st[5] * st[5] * sinTheta2) + (0.75 * MassPole2 * cosTheta2 * (temp2 + gSinTheta2))
	mi1 := MassPole1 * (1 - (0.75 * cosTheta1 * cosTheta1))
	mi2 := MassPole2 * (1 - (0.75 * cosTheta2 * cosTheta2))

	//fmt.Printf("%f -> %f\n", action, force)

	derivs[1] = (force + fi1 + fi2) / (mi1 + mi2 + MassCart)
	derivs[3] = -0.75 * (derivs[1]*cosTheta1 + gSinTheta1 + temp1) / Length1
	derivs[5] = -0.75 * (derivs[1]*cosTheta2 + gSinTheta2 + temp2) / Length2
}

func (p *CartPole) rk4(f float64, y, dydx [6]float64, yout *[6]float64, tau float64) {
	var yt, dym, dyt [6]float64
	hh := tau * 0.5
	h6 := tau / 6.0
	for i := 0; i <= 5; i++ {
		yt[i] = y[i] + hh*dydx[i]
	}
	p.step(f, yt, &dyt)

	dyt[0] = yt[1]
	dyt[2] = yt[3]
	dyt[4] = yt[5]
	for i := 0; i <= 5; i++ {
		yt[i] = y[i] + hh*dyt[i]
	}
	p.step(f, yt, &dym)

	dym[0] = yt[1]
	dym[2] = yt[3]
	dym[4] = yt[5]
	for i := 0; i <= 5; i++ {
		yt[i] = y[i] + tau*dym[i]
		dym[i] += dyt[i]
	}
	p.step(f, yt, &dyt)

	dyt[0] = yt[1]
	dyt[2] = yt[3]
	dyt[4] = yt[5]
	for i := 0; i <= 5; i++ {
		yout[i] = y[i] + h6*(dydx[i]+dyt[i]+2.0*dym[i])
	}
}

// Check if simulation goes outside of bounds
func (p *CartPole) outsideBounds() bool {
	const failureAngle = thirtySixDegrees

	return p.state[0] < -2.4 ||
		p.state[0] > 2.4 ||
		p.state[2] < -failureAngle ||
		p.state[2] > failureAngle ||
		p.state[4] < -failureAngle ||
		p.state[4] > failureAngle
}

func (p *CartPole) resetState() {
	if p.isMarkov {
		// Clear all fitness records
		p.cartPosSum = 0.0
		p.cartVelocitySum = 0.0
		p.polePosSum = 0.0
		p.poleVelocitySum = 0.0

		p.state[0], p.state[1], p.state[2], p.state[3], p.state[4], p.state[5] = 0, 0, 0, 0, 0, 0
	} else if !p.generalizationTest {
		// The long run non-markov test
		p.state[0], p.state[1], p.state[3], p.state[4], p.state[5] = 0, 0, 0, 0, 0
		p.state[2] = math.Pi / 180.0 // one_degree
	}
	p.balancedTimeSteps = 0 // Always count # of balanced time steps
}
