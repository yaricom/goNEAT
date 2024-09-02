package pole2

import (
	"fmt"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"sort"
)

// EvaluateOrganismGeneralization
// The best individual (i.e. the one with the highest fitness value) of every generation is tested for
// its ability to balance the system for a longer time period. If a potential solution passes this test
// by keeping the system balanced for 100’000 time steps, the so-called generalization score(GS) of this
// particular individual is calculated. This score measures the potential of a controller to balance the
// system starting from different initial conditions. It's calculated with a series of experiments, running
// over 1000 time steps, starting from 625 different initial conditions.
// The initial conditions are chosen by assigning each value of the set Ω = [0.05 0.25 0.5 0.75 0.95] to
// each of the states x, ∆x/∆t, θ1 and ∆θ1/∆t, scaled to the range of the variables.The short pole angle θ2
// and its angular velocity ∆θ2/∆t are set to zero. The GS is then defined as the number of successful runs
// from the 625 initial conditions and an individual is defined as a solution if it reaches a generalization
// score of 200 or more.
func EvaluateOrganismGeneralization(species []*genetics.Species, cartPole *CartDoublePole, actionType ActionType) (*genetics.Organism, error) {
	// Sort the species by max organism fitness in descending order - the highest fitness first
	sortedSpecies := make([]*genetics.Species, len(species))
	copy(sortedSpecies, species)
	sort.Sort(sort.Reverse(genetics.ByOrganismFitness(sortedSpecies)))

	// First update what is checked and unchecked
	var currSpecies *genetics.Species
	for _, currSpecies = range sortedSpecies {
		maxFitness, _ := currSpecies.ComputeMaxAndAvgFitness()
		if maxFitness > currSpecies.MaxFitnessEver {
			currSpecies.IsChecked = false
		} else {
			currSpecies.IsChecked = true
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
	championPhenotype, err := champion.Phenotype()
	if err != nil {
		return nil, err
	}

	// Now check to make sure the champion can do 100'000 evaluations
	cartPole.nonMarkovLong = true
	cartPole.generalizationTest = false

	longRunPassed, err := OrganismEvaluate(champion, cartPole, actionType)
	if err != nil {
		return nil, err
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
						if _, err = championPhenotype.Flush(); err != nil {
							return nil, err
						}

						if generalized, err := OrganismEvaluate(champion, cartPole, actionType); generalized {
							generalizationScore++

							if neat.LogLevel == neat.LogLevelDebug {
								neat.DebugLog(
									fmt.Sprintf("x: %f, xv: %f, t1: %f, t2: %f, angle: %f\n",
										cartPole.state[0], cartPole.state[1],
										cartPole.state[2], cartPole.state[4], thirtySixDegrees))
							}
						} else if err != nil {
							return nil, err
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

	return champion, nil
}
