package pole

import (
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat/network"
	"fmt"
	"github.com/yaricom/goNEAT/neat"
	"math"
	"github.com/yaricom/goNEAT/neat/genetics"
	"os"
	"sort"
)

const thirty_six_degrees = 36 * math.Pi / 180.0

// The maximal number of time steps for Markov experiment
const markov_max_steps = 100000
// The maximal number of time steps for Non-Markov long run
const non_markov_long_max_steps = 100000
// The maximal number of time steps for Non-Markov generalization run
const non_markov_generalization_max_steps = 1000


// The double pole-balancing experiment both Markov and non-Markov versions
type CartDoublePoleGenerationEvaluator struct {
	// The output path to store execution results
	OutputPath string
	// The flag to indicate whether to apply Markov evaluation variant
	Markov     bool

	// The flag to indicate whether to use continuous activation or discrete
	ActionType experiments.ActionType
}

// The structure to describe cart pole emulation
type CartPole struct {
	// The flag to indicate that we are executing Markov experiment setup (known velocities information)
	isMarkov            bool
	// Flag that we are looking at the champion in Non-Markov experiment
	nonMarkovLong       bool
	// Flag that we are testing champion's generalization
	generalizationTest  bool

	// The state of the system (x, ∆x/∆t, θ1, ∆θ1/∆t, θ2, ∆θ2/∆t)
	state               [6]float64

	// The number of balanced time steps passed for current organism evaluation
	balanced_time_steps int

	jiggleStep          [1000]float64

	// Queues used for Gruau's fitness which damps oscillations

	cartpos_sum         float64
	cartv_sum           float64
	polepos_sum         float64
	polev_sum           float64
}

// Perform evaluation of one epoch on double pole balancing
func (ex CartDoublePoleGenerationEvaluator) GenerationEvaluate(pop *genetics.Population, epoch *experiments.Generation, context *neat.NeatContext) (err error) {
	cartPole := newCartPole(ex.Markov)

	cartPole.nonMarkovLong = false
	cartPole.generalizationTest = false

	// Evaluate each organism on a test
	for _, org := range pop.Organisms {
		winner, err := ex.orgEvaluate(org, cartPole)
		if err != nil {
			return err
		}

		if winner && (epoch.Best == nil || org.Fitness > epoch.Best.Fitness){
			// This will be winner in Markov case
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize * epoch.Id + org.Genotype.Id
			epoch.Best = org
			org.IsWinner = true
		}
	}

	// Check for winner in Non-Markov case
	if !ex.Markov {
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
		sorted_species := make([]*genetics.Species, len(pop.Species))
		copy(sorted_species, pop.Species)
		sort.Sort(sort.Reverse(genetics.ByOrganismFitness(sorted_species)))

		// First update what is checked and unchecked
		var curr_species *genetics.Species
		for _, curr_species = range sorted_species {
			max, _ := curr_species.ComputeMaxAndAvgFitness()
			if max > curr_species.MaxFitnessEver {
				curr_species.IsChecked = false
			}
		}

		// Now find first (most fit) species that is unchecked
		curr_species = nil
		for _, curr_species = range sorted_species {
			if !curr_species.IsChecked {
				break
			}
		}
		if curr_species == nil {
			curr_species = sorted_species[0]
		}

		// Remember it was checked
		curr_species.IsChecked = true

		// the organism champion
		champion := curr_species.FindChampion()
		champion_fitness := champion.Fitness

		// Now check to make sure the champion can do 100'000 evaluations
		cartPole.nonMarkovLong = true
		cartPole.generalizationTest = false

		longRunPassed, err := ex.orgEvaluate(champion, cartPole)
		if err != nil {
			return err
		}
		if longRunPassed {

			// the champion passed non-Markov long test, start generalization
			cartPole.nonMarkovLong = false
			cartPole.generalizationTest = true

			// Given that the champion passed long run test, now run it on generalization tests running
			// over 1'000 time steps, starting from 625 different initial conditions
			state_vals := [5]float64{0.05, 0.25, 0.5, 0.75, 0.95}
			generalization_score := 0
			for s0c := 0; s0c < 5; s0c++ {
				for s1c := 0; s1c < 5; s1c++ {
					for s2c := 0; s2c < 5; s2c++ {
						for s3c := 0; s3c < 5; s3c++ {
							cartPole.state[0] = state_vals[s0c] * 4.32 - 2.16
							cartPole.state[1] = state_vals[s1c] * 2.70 - 1.35
							cartPole.state[2] = state_vals[s2c] * 0.12566304 - 0.06283152 // 0.06283152 = 3.6 degrees
							cartPole.state[3] = state_vals[s3c] * 0.30019504 - 0.15009752 // 0.15009752 = 8.6 degrees
							// The short pole angle and its angular velocity are set to zero.
							cartPole.state[4] = 0.0
							cartPole.state[5] = 0.0

							// The champion needs to be flushed here because it may have
							// leftover activation from its last test run that could affect
							// its recurrent memory
							champion.Phenotype.Flush()

							if generalized, err := ex.orgEvaluate(champion, cartPole); generalized {
								generalization_score++

								if neat.LogLevel == neat.LogLevelDebug {
									neat.DebugLog(
										fmt.Sprintf("x: % f, xv: % f, t1: % f, t2: % f, angle: % f\n",
										cartPole.state[0], cartPole.state[1],
										cartPole.state[2], cartPole.state[4], thirty_six_degrees))
								}
							} else if err != nil {
								return err
							}
						}
					}
				}
			}

			if generalization_score >= 200 {
				// The generalization test winner
				neat.InfoLog(
					fmt.Sprintf("The non-Markov champion found! (Generalization Score = %d)",
						generalization_score))
				champion.Fitness = float64(generalization_score)
				champion.IsWinner = true
				epoch.Solved = true
				epoch.WinnerNodes = len(champion.Genotype.Nodes)
				epoch.WinnerGenes = champion.Genotype.Extrons()
				epoch.WinnerEvals = context.PopSize * epoch.Id + champion.Genotype.Id
				epoch.Best = champion
			} else {
				neat.InfoLog("The non-Markov champion unable to generalize")
				champion.Fitness = champion_fitness // Restore the champ's fitness
				champion.IsWinner = false
			}
		} else {
			neat.InfoLog("The non-Markov champion missed the 100'000 run test")
			champion.Fitness = champion_fitness // Restore the champ's fitness
			champion.IsWinner = false
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
				org_path := fmt.Sprintf("%s/%s_%.1f_%d-%d", experiments.OutDirForTrial(ex.OutputPath, epoch.TrialId),
					"pole2_winner", org.Fitness, org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
				file, err := os.Create(org_path)
				if err != nil {
					neat.ErrorLog(fmt.Sprintf("Failed to dump winner organism genome, reason: %s\n", err))
				} else {
					org.Genotype.Write(file)
					neat.InfoLog(fmt.Sprintf("Generation #%d winner %d dumped to: %s\n", epoch.Id, org.Genotype.Id, org_path))
				}
				break
			}
		}
	}

	return err
}

// This methods evaluates provided organism for cart double pole-balancing task
func (ex *CartDoublePoleGenerationEvaluator) orgEvaluate(organism *genetics.Organism, cartPole *CartPole) (winner bool, err error) {
	// Try to balance a pole now
	organism.Fitness, err = cartPole.evalNet(organism.Phenotype, ex.ActionType)
	if err != nil {
		return false, err
	}

	if neat.LogLevel == neat.LogLevelDebug {
		neat.DebugLog(fmt.Sprintf("Organism #%3d\tfitness: %f", organism.Genotype.Id, organism.Fitness))
	}

	// DEBUG CHECK if organism is damaged
	if !(cartPole.nonMarkovLong && cartPole.generalizationTest) && organism.CheckChampionChildDamaged() {
		neat.WarnLog(fmt.Sprintf("ORGANISM DAMAGED:\n%s", organism.Genotype))
	}

	// Decide if its a winner, in Markov Case
	if cartPole.isMarkov {
		if organism.Fitness >= markov_max_steps {
			winner = true
			organism.Fitness = 1.0
			organism.Error = 0.0
		} else {
			// we use linear scale
			organism.Error = (markov_max_steps - organism.Fitness) / markov_max_steps
			organism.Fitness = 1.0 - organism.Error
		}
	} else if cartPole.nonMarkovLong {
		// if doing the long test non-markov
		if organism.Fitness >= non_markov_long_max_steps {
			winner = true
		}
	} else if cartPole.generalizationTest {
		if organism.Fitness >= non_markov_generalization_max_steps {
			winner = true
		}
	} else {
		winner = false
	}
	return winner, err
}


// If markov is false, then velocity information will be withheld from the network population (non-Markov)
func newCartPole(markov bool) *CartPole {
	return &CartPole {
		isMarkov: markov,
	}
}

func (cp *CartPole)evalNet(net *network.Network, actionType experiments.ActionType) (steps float64, err error) {
	non_markov_max := non_markov_generalization_max_steps
	if cp.nonMarkovLong {
		non_markov_max = non_markov_long_max_steps
	}



	cp.resetState()

	if cp.isMarkov {
		input := make([]float64, 7)
		for steps = 0; steps < markov_max_steps; steps++ {
			input[0] = (cp.state[0] + 2.4) / 4.8
			input[1] = (cp.state[1] + 1.0) / 2.0
			input[2] = (cp.state[2] + thirty_six_degrees) / (thirty_six_degrees * 2.0)//0.52
			input[3] = (cp.state[3] + 1.0) / 2.0
			input[4] = (cp.state[4] + thirty_six_degrees) / (thirty_six_degrees * 2.0)//0.52
			input[5] = (cp.state[5] + 1.0) / 2.0
			input[6] = 0.5

			net.LoadSensors(input)

			/*-- activate the network based on the input --*/
			if res, err := net.Activate(); !res {
				//If it loops, exit returning only fitness of 1 step
				neat.DebugLog(fmt.Sprintf("Failed to activate Network, reason: %s", err))
				return 1.0, nil
			}
			action := net.Outputs[0].Activation
			if actionType == experiments.DiscreteAction {
				// make action values discrete
				if action < 0.5 {
					action = 0
				} else {
					action = 1
				}
			}
			cp.performAction(action, steps)

			if cp.outsideBounds() {
				// if failure stop it now
				break;
			}

			//fmt.Printf("x: % f, xv: % f, t1: % f, t2: % f, angle: % f\n", cp.state[0], cp.state[1], cp.state[2], cp.state[4], thirty_six_degrees)
		}
		return steps, nil
	} else {
		input := make([]float64, 4)
		// The non Markov case
		for steps = 0; steps < float64(non_markov_max); steps++ {
			input[0] = cp.state[0] / 4.8
			input[1] = cp.state[2] / 0.52
			input[2] = cp.state[4] / 0.52
			input[3] = 1.0

			err = net.LoadSensors(input)
			if err != nil {
				return 0, err
			}

			/*-- activate the network based on the input --*/
			if res, err := net.Activate(); !res {
				// If it loops, exit returning only fitness of 1 step
				neat.WarnLog(fmt.Sprintf("Failed to activate Network, reason: %s", err))
				return 0.0001, err
			}

			action := net.Outputs[0].Activation
			if actionType == experiments.DiscreteAction {
				// make action values discrete
				if action < 0.5 {
					action = 0
				} else {
					action = 1
				}
			}
			cp.performAction(action, steps)
			if cp.outsideBounds() {
				//fmt.Printf("x: % f, xv: % f, t1: % f, t2: % f, angle: % f, steps: %f\n",
				//	cp.state[0], cp.state[1], cp.state[2], cp.state[4], thirty_six_degrees, steps)
				// if failure stop it now
				break;
			}


		}
		/*-- If we are generalizing we just need to balance it for a while --*/
		if cp.generalizationTest {
			return float64(cp.balanced_time_steps), nil
		}

		// Sum last 100
		jiggle_total := 0.0
		if steps >= 100.0 && !cp.nonMarkovLong {
			// Adjust for array bounds and count
			for count := int(steps - 100.0); count < int(steps); count++ {
				jiggle_total += cp.jiggleStep[count]
			}
		}
		if !cp.nonMarkovLong {
			var non_markov_fitness float64
			if cp.balanced_time_steps >= 100 {
				// F = 0.1f1 + 0.9f2
				non_markov_fitness = 0.1 * float64(cp.balanced_time_steps) / 1000.0 + 0.9 * 0.75 / float64(jiggle_total)
			} else {
				// F = t / 1000
				non_markov_fitness = 0.1 * float64(cp.balanced_time_steps) / 1000.0
			}
			if neat.LogLevel == neat.LogLevelDebug {
				neat.DebugLog(fmt.Sprintf("Balanced time steps: %d, jiggle: %f ***\n",
					cp.balanced_time_steps, jiggle_total))
			}
			return non_markov_fitness, nil
		} else {
			return steps, nil
		}
	}
}

func (cp *CartPole) performAction(action, step_num float64) {
	const TAU = 0.01 // ∆t = 0.01s

	/*--- Apply action to the simulated cart-pole ---*/
	// Runge-Kutta 4th order integration method
	var dydx[6]float64
	for i := 0; i < 2; i++ {
		dydx[0] = cp.state[1]
		dydx[2] = cp.state[3]
		dydx[4] = cp.state[5]
		cp.step(action, cp.state, &dydx)
		cp.rk4(action, cp.state, dydx, &cp.state, TAU)
	}

	// Record this state
	cp.cartpos_sum += math.Abs(cp.state[0])
	cp.cartv_sum += math.Abs(cp.state[1]);
	cp.polepos_sum += math.Abs(cp.state[2]);
	cp.polev_sum += math.Abs(cp.state[3]);

	if step_num < 1000 {
		cp.jiggleStep[int(step_num)] = math.Abs(cp.state[0]) + math.Abs(cp.state[1]) + math.Abs(cp.state[2]) + math.Abs(cp.state[3])
	}
	if !cp.outsideBounds() {
		cp.balanced_time_steps++
	}
}

func (cp *CartPole) step(action float64, st[6]float64, derivs *[6]float64) {
	const MUP = 0.000002
	const GRAVITY = -9.8
	const FORCE_MAG = 10.0       // [N]
	const MASS_CART = 1.0        // [kg]

	const MASS_POLE_1 = 1.0      // [kg]
	const LENGTH_1 = 0.5         // [m] - actually half the first pole's length

	const LENGTH_2 = 0.05        // [m] - actually half the second pole's length
	const MASS_POLE_2 = 0.1      // [kg]

	force := (action - 0.5) * FORCE_MAG * 2.0
	cos_theta_1 := math.Cos(st[2])
	sin_theta_1 := math.Sin(st[2])
	g_sin_theta_1 := GRAVITY * sin_theta_1
	cos_theta_2 := math.Cos(st[4])
	sin_theta_2 := math.Sin(st[4])
	g_sin_theta_2 := GRAVITY * sin_theta_2

	ml_1 := LENGTH_1 * MASS_POLE_1
	ml_2 := LENGTH_2 * MASS_POLE_2
	temp_1 := MUP * st[3] / ml_1
	temp_2 := MUP * st[5] / ml_2
	fi_1 := (ml_1 * st[3] * st[3] * sin_theta_1) + (0.75 * MASS_POLE_1 * cos_theta_1 * (temp_1 + g_sin_theta_1))
	fi_2 := (ml_2 * st[5] * st[5] * sin_theta_2) + (0.75 * MASS_POLE_2 * cos_theta_2 * (temp_2 + g_sin_theta_2))
	mi_1 := MASS_POLE_1 * (1 - (0.75 * cos_theta_1 * cos_theta_1))
	mi_2 := MASS_POLE_2 * (1 - (0.75 * cos_theta_2 * cos_theta_2))

	//fmt.Printf("%f -> %f\n", action, force)

	derivs[1] = (force + fi_1 + fi_2) / (mi_1 + mi_2 + MASS_CART)
	derivs[3] = -0.75 * (derivs[1] * cos_theta_1 + g_sin_theta_1 + temp_1) / LENGTH_1
	derivs[5] = -0.75 * (derivs[1] * cos_theta_2 + g_sin_theta_2 + temp_2) / LENGTH_2
}

func (cp *CartPole) rk4(f float64, y, dydx [6]float64, yout *[6]float64, tau float64) {
	var yt, dym, dyt [6]float64
	hh := tau * 0.5
	h6 := tau / 6.0
	for i := 0; i <= 5; i++ {
		yt[i] = y[i] + hh * dydx[i]
	}
	cp.step(f, yt, &dyt)

	dyt[0] = yt[1]
	dyt[2] = yt[3]
	dyt[4] = yt[5]
	for i := 0; i <= 5; i++ {
		yt[i] = y[i] + hh * dyt[i]
	}
	cp.step(f, yt, &dym)

	dym[0] = yt[1]
	dym[2] = yt[3]
	dym[4] = yt[5]
	for i := 0; i <= 5; i++ {
		yt[i] = y[i] + tau * dym[i]
		dym[i] += dyt[i]
	}
	cp.step(f, yt, &dyt)

	dyt[0] = yt[1]
	dyt[2] = yt[3]
	dyt[4] = yt[5]
	for i := 0; i <= 5; i++ {
		yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i])
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

		cp.state[0], cp.state[1], cp.state[2], cp.state[3], cp.state[4], cp.state[5] = 0, 0, 0, 0, 0, 0
	} else if !cp.generalizationTest {
		// The long run non-markov test
		cp.state[0], cp.state[1], cp.state[3], cp.state[4], cp.state[5] = 0, 0, 0, 0, 0
		cp.state[2] = math.Pi / 180.0 // one_degree
	}
	cp.balanced_time_steps = 0 // Always count # of balanced time steps
}


