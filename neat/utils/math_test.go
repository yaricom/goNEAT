package utils

import (
	"testing"
	"math/rand"
)

func TestSingleRouletteThrow(t *testing.T) {
	rand.Seed(42)
	probs := []float64{.1, .2, .4, .15, .15}

	hist := make([]float64, len(probs))
	runs := 10000
	for i := 0; i < runs; i++ {
		index := SingleRouletteThrow(probs)
		if index < 0 || index >= len(probs) {
			t.Error("invalid segment index", index)
			return
		}
		// increment histogram to check distribution quality
		hist[index] += 1
	}
	t.Log(hist)
}
