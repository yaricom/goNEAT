package pole

import (
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat/network"
	"fmt"
	"github.com/yaricom/goNEAT/neat"
	"math"
	"github.com/yaricom/goNEAT/neat/genetics"
	"os"
)

const thirty_six_degrees = 36 * math.Pi / 180.0


// The double pole-balancing experiment both Markov and non-Markov versions
type CartDoublePoleEpochEvaluator struct {
	// The output path to store execution results
	OutputPath string
	// The flag to indicate whether to apply Markov evaluation variant
	Markov     bool

	// The currently evaluating cart pole
	cartPole   *CartPole
}

// The structure to describe cart pole emulation
type CartPole struct {
	// The maximal fitness
	maxFitness         float64
	// The flag to indicate whether to apply Markovian evaluation variant
	isMarkov           bool
	// Flag that we are looking at the champ
	nonMarkovLong      bool
	// Flag we are testing champ's generalization
	generalizationTest bool
	// The state of the system
	state              [6]float64

	jiggleStep         [1000]float64

	length2            float64
	massPole2          float64
	minInc             float64
	poleInc            float64
	massInc            float64

	// Queues used for Gruau's fitness which damps oscillations
	balanced_sum       int
	cartpos_sum        float64
	cartv_sum          float64
	polepos_sum        float64
	polev_sum          float64
}

func (ev *CartDoublePoleEpochEvaluator) TrialRunStarted(trial *experiments.Trial) {
	ev.cartPole = newCartPole(ev.Markov)
}

// Perform evaluation of one epoch on double pole balancing
func (ex *CartDoublePoleEpochEvaluator) EpochEvaluate(pop *genetics.Population, epoch *experiments.Epoch, context *neat.NeatContext) (err error) {
	ex.cartPole.nonMarkovLong = false
	ex.cartPole.generalizationTest = false

	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		res := ex.orgEvaluate(org)

		if res {
			// This will be winner in Markov case
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize * epoch.Id + org.Genotype.Id
			epoch.Best = org
			break // we have winner
		}
	}

	// Check for winner in Non-Markov case
	if !ex.Markov {
		// TODO: finish implementation
	}


	// Fill statistics about current epoch
	epoch.FillPopulationStatistics(pop)

	// Only print to file every print_every generations
	if epoch.Solved || epoch.Id % context.PrintEvery == 0 {
		pop_path := fmt.Sprintf("%s/gen_%d", ex.OutputPath, epoch.Id)
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
				org_path := fmt.Sprintf("%s/%s", ex.OutputPath, "xor_winner")
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
	} else {
		// Move to the next epoch if failed to find winner
		neat.DebugLog(">>>>> start next generation")
		_, err = pop.Epoch(epoch.Id + 1, context)
	}

	return err
}

// This methods evaluates provided organism for cart double pole-balancing task
func (ex *CartDoublePoleEpochEvaluator) orgEvaluate(organism *genetics.Organism) bool {
	// Try to balance a pole now
	organism.Fitness = ex.cartPole.evalNet(organism.Phenotype)

	// DEBUG CHECK if organism is damaged
	if !(ex.cartPole.nonMarkovLong && ex.cartPole.generalizationTest) && organism.CheckChampionChildDamaged() {
		neat.WarnLog(fmt.Sprintf("ORGANISM DAMAGED:\n%s", organism.Genotype))
	}

	// Decide if its a winner, in Markov Case
	if ex.cartPole.isMarkov {
		if organism.Fitness >= ex.cartPole.maxFitness {
			organism.IsWinner = true
		}
	} else if ex.cartPole.nonMarkovLong {
		// if doing the long test non-markov
		if organism.Fitness >= 99999 {
			organism.IsWinner = true
		}
	} else if ex.cartPole.generalizationTest {
		if organism.Fitness >= 999 {
			organism.IsWinner = true
		}
	} else {
		organism.IsWinner = false
	}
	return organism.IsWinner
}


// If markov is false, then velocity information will be withheld from the network population (non-Markov)
func newCartPole(markov bool) *CartPole {
	return &CartPole{
		maxFitness: 100000,
		isMarkov: markov,
		minInc: 0.001,
		poleInc: 0.05,
		massInc: 0.01,
		length2: 0.05,
		massPole2: 0.01,
	}
}

func (cp *CartPole)evalNet(net *network.Network) (steps float64) {
	non_markov_max := 1000.0
	if cp.nonMarkovLong {
		non_markov_max = 100000.0
	} else if cp.generalizationTest {
		non_markov_max = 1000.0
	}

	input := make([]float64, 7)

	cp.resetState()

	if cp.isMarkov {
		for ; steps < cp.maxFitness; steps++ {
			input[0] = cp.state[0] / 4.8
			input[1] = cp.state[1] / 2
			input[2] = cp.state[2] / 0.52
			input[3] = cp.state[3] / 2
			input[4] = cp.state[4] / 0.52
			input[5] = cp.state[5] / 2
			input[6] = 0.5

			net.LoadSensors(input)

			/*-- activate the network based on the input --*/
			if res, err := net.Activate(); !res {
				//If it loops, exit returning only fitness of 1 step
				neat.DebugLog(fmt.Sprintf("Failed to activate Network, reason: %s", err))
				return 1.0
			}
			output := net.Outputs[0].Activation
			cp.performAction(output, steps)

			if cp.outsideBounds() {
				// if failure stop it now
				break;
			}
		}
		return steps
	} else {
		// The non Markov case
		for ; steps < non_markov_max; steps++ {
			input[0] = cp.state[0] / 4.8
			input[1] = cp.state[2] / 0.52
			input[2] = cp.state[4] / 0.52
			input[3] = 0.5

			net.LoadSensors(input)

			/*-- activate the network based on the input --*/
			if res, err := net.Activate(); !res {
				//If it loops, exit returning only fitness of 1 step
				neat.DebugLog(fmt.Sprintf("Failed to activate Network, reason: %s", err))
				return 0.0001
			}

			output := net.Outputs[0].Activation
			cp.performAction(output, steps)

			if cp.outsideBounds() {
				// if failure stop it now
				break;
			}
		}
		/*-- If we are generalizing we just need to balance it a while --*/
		if cp.generalizationTest {
			return float64(cp.balanced_sum)
		}

		// Sum last 100
		jiggle_total := 0.0
		if steps > 100.0 && !cp.nonMarkovLong {
			// Adjust for array bounds and count
			for count := int(steps - 99.0 - 2.0); count <= int(steps - 2.0); count++ {
				jiggle_total += cp.jiggleStep[count]
			}
		}
		if !cp.nonMarkovLong {
			var non_markov_fitness float64
			if cp.balanced_sum > 100 {
				non_markov_fitness = 0.1 * float64(cp.balanced_sum) / 1000.0 + 0.9 * 0.75 / float64(jiggle_total)
			} else {
				non_markov_fitness = 0.1 * float64(cp.balanced_sum) / 1000.0
			}
			if neat.LogLevel == neat.LogLevelDebug {
				neat.DebugLog(fmt.Sprintf("Balanced: %d jiggle: %d ***\n", cp.balanced_sum, jiggle_total))
			}
			return non_markov_fitness
		} else {
			return steps
		}
	}
}

func (cp *CartPole) performAction(output, step_num float64) {
	const TAU = 0.01

	var dydx [6]float64
	/*--- Apply action to the simulated cart-pole ---*/
	// Runge-Kutta 4th order integration method
	for i := 0; i < 2; i++ {
		dydx[0] = cp.state[1];
		dydx[2] = cp.state[3];
		dydx[4] = cp.state[5];
		cp.step(output, dydx);
		cp.rk4(output, dydx, TAU);
	}
	// Record this state
	cp.cartpos_sum += math.Abs(cp.state[0])
	cp.cartv_sum += math.Abs(cp.state[1]);
	cp.polepos_sum += math.Abs(cp.state[2]);
	cp.polev_sum += math.Abs(cp.state[3]);

	if step_num <= 1000 {
		cp.jiggleStep[int(step_num) - 1] = math.Abs(cp.state[0]) + math.Abs(cp.state[1]) + math.Abs(cp.state[2]) + math.Abs(cp.state[3])
	}
	if !cp.outsideBounds() {
		cp.balanced_sum++
	}
}

func (cp *CartPole) step(action float64, derivs [6]float64) {
	const MUP = 0.000002
	const GRAVITY = -9.8
	const MASSCART = 1.0
	const MASSPOLE_1 = 0.1
	const LENGTH_1 = 0.5 // actually half the pole's length
	const FORCE_MAG = 10.0

	var force, cos_theta_1, cos_theta_2, sin_theta_1, sin_theta_2,
	g_sin_theta_1, g_sin_theta_2, temp_1, temp_2, ml_1, ml_2, fi_1, fi_2, mi_1, mi_2 float64

	force = (action - 0.5) * FORCE_MAG * 2
	cos_theta_1 = math.Cos(cp.state[2])
	sin_theta_1 = math.Sin(cp.state[2])
	g_sin_theta_1 = GRAVITY * sin_theta_1
	cos_theta_2 = math.Cos(cp.state[4])
	sin_theta_2 = math.Sin(cp.state[4])
	g_sin_theta_2 = GRAVITY * sin_theta_2

	ml_1 = LENGTH_1 * MASSPOLE_1
	ml_2 = cp.length2 * cp.massPole2
	temp_1 = MUP * cp.state[3] / ml_1
	temp_2 = MUP * cp.state[5] / ml_2
	fi_1 = (ml_1 * cp.state[3] * cp.state[3] * sin_theta_1) + (0.75 * MASSPOLE_1 * cos_theta_1 * (temp_1 + g_sin_theta_1))
	fi_2 = (ml_2 * cp.state[5] * cp.state[5] * sin_theta_2) + (0.75 * cp.massPole2 * cos_theta_2 * (temp_2 + g_sin_theta_2))
	mi_1 = MASSPOLE_1 * (1 - (0.75 * cos_theta_1 * cos_theta_1))
	mi_2 = cp.massPole2 * (1 - (0.75 * cos_theta_2 * cos_theta_2))

	derivs[1] = (force + fi_1 + fi_2) / (mi_1 + mi_2 + MASSCART)

	derivs[3] = -0.75 * (derivs[1] * cos_theta_1 + g_sin_theta_1 + temp_1) / LENGTH_1
	derivs[5] = -0.75 * (derivs[1] * cos_theta_2 + g_sin_theta_2 + temp_2) / cp.length2
}

func (cp *CartPole) rk4(f float64, dydx [6]float64, tau float64) {
	var dym, dyt, yt [6]float64
	hh := tau * 0.5
	h6 := tau / 6.0
	for i := 0; i <= 5; i++ {
		yt[i] = cp.state[i] + hh * dydx[i]
	}
	cp.step(f, dyt)

	dyt[0] = yt[1]
	dyt[2] = yt[3]
	dyt[4] = yt[5]
	for i := 0; i <= 5; i++ {
		yt[i] = cp.state[i] + hh * dyt[i]
	}
	cp.step(f, dym)

	dym[0] = yt[1]
	dym[2] = yt[3]
	dym[4] = yt[5]
	for i := 0; i <= 5; i++ {
		yt[i] = cp.state[i] + tau * dym[i]
		dym[i] += dyt[i]
	}
	cp.step(f, dyt)

	dyt[0] = yt[1]
	dyt[2] = yt[3]
	dyt[4] = yt[5]
	for i := 0; i <= 5; i++ {
		cp.state[i] = cp.state[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i])
	}
}

// Check if simulation goes outside of bounds
func (cp *CartPole) outsideBounds() bool {
	const failureAngle = thirty_six_degrees

	return cp.state[0] < -2.4 ||
		cp.state[0] > 2.4 ||
		cp.state[2] < -failureAngle ||
		cp.state[2] > failureAngle ||
		cp.state[4] < -failureAngle ||
		cp.state[4] > failureAngle
}

func (cp *CartPole)resetState() {
	if cp.isMarkov {
		// Clear all fitness records
		cp.cartpos_sum = 0.0
		cp.cartv_sum = 0.0
		cp.polepos_sum = 0.0
		cp.polev_sum = 0.0
	}
	cp.balanced_sum = 0 //Always count # balanced
	if cp.generalizationTest {
		cp.state[0], cp.state[1], cp.state[3], cp.state[4], cp.state[5] = 0, 0, 0, 0, 0
		cp.state[2] = math.Pi / 180.0 // one_degree
	} else {
		cp.state[4], cp.state[5] = 0, 0
	}
}


