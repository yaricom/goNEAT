// Package pole2 provides definition of the two pole balancing experiment.
// In this experiment we will try to teach RF model of balancing of two poles placed on the moving cart.
package pole2

import (
	"fmt"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"github.com/yaricom/goNEAT/v4/neat/network"
	"math"
)

const thirtySixDegrees = 36 * math.Pi / 180.0

// The maximal number of time steps for Markov experiment
const markovMaxSteps = 100000

// The maximal number of time steps for Non-Markov long run
const nonMarkovLongMaxSteps = 100000

// The maximal number of time steps for Non-Markov generalization run
const nonMarkovGeneralizationMaxSteps = 1000

// ActionType The type of action to be applied to environment
type ActionType byte

// The supported action types
const (
	// ContinuousAction The continuous action type meaning continuous values to be applied to environment
	ContinuousAction ActionType = iota
	// DiscreteAction The discrete action assumes that there are only discrete values of action (e.g. 0, 1)
	DiscreteAction
)

// CartDoublePole The structure to describe cart pole emulation
type CartDoublePole struct {
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

// NewCartPole If markov is false, then velocity information will be withheld from the network population (non-Markov)
func NewCartPole(markov bool) *CartDoublePole {
	return &CartDoublePole{
		isMarkov: markov,
	}
}

func (p *CartDoublePole) evalNet(net *network.Network, actionType ActionType) (steps float64, err error) {
	nonMarkovMax := nonMarkovGeneralizationMaxSteps
	if p.nonMarkovLong {
		nonMarkovMax = nonMarkovLongMaxSteps
	}

	p.resetState()

	netDepth, err := net.MaxActivationDepthWithCap(0) // The max depth of the network to be activated
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

func (p *CartDoublePole) performAction(action, stepNum float64) {
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

func (p *CartDoublePole) step(action float64, st [6]float64, derivs *[6]float64) {
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

func (p *CartDoublePole) rk4(f float64, y, dydx [6]float64, yout *[6]float64, tau float64) {
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
func (p *CartDoublePole) outsideBounds() bool {
	const failureAngle = thirtySixDegrees

	return p.state[0] < -2.4 ||
		p.state[0] > 2.4 ||
		p.state[2] < -failureAngle ||
		p.state[2] > failureAngle ||
		p.state[4] < -failureAngle ||
		p.state[4] > failureAngle
}

func (p *CartDoublePole) resetState() {
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

// OrganismEvaluate method evaluates fitness of the organism for cart double pole-balancing task
func OrganismEvaluate(organism *genetics.Organism, cartPole *CartDoublePole, actionType ActionType) (winner bool, err error) {
	// Try to balance a pole now
	phenotype, err := organism.Phenotype()
	if err != nil {
		return false, err
	}
	organism.Fitness, err = cartPole.evalNet(phenotype, actionType)
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
