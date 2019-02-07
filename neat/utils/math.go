package utils

import (
	"math/rand"
)

// Returns subsequent random positive or negative integer value (1 or -1) to randomize value sign
func RandSign() int32 {
	v := rand.Int()
	if (v % 2) == 0 {
		return -1
	} else {
		return 1
	}
}

// Performs a single thrown onto a roulette wheel where the wheel's space is unevenly divided.
// The probability that a segment will be selected is given by that segment's value in the probabilities array.
// Returns segment index or -1 if something goes awfully wrong
func SingleRouletteThrow(probabilities []float64) int  {
	total := 0.0

	// collect all probabilities
	for _, v := range probabilities {
		total += v
	}

	// throw the ball and collect result
	throwValue := rand.Float64() * total

	accumulator := 0.0
	for i, v := range probabilities {
		accumulator += v
		if throwValue <= accumulator {
			return i
		}
	}
	return -1
}
