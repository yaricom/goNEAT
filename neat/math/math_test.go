package math

import (
	"math/rand"
	"testing"
)

func TestSingleRouletteThrow(t *testing.T) {
	rand.Seed(42)
	probabilities := []float64{.1, .2, .4, .15, .15}

	hist := make([]float64, len(probabilities))
	runs := 10000
	for i := 0; i < runs; i++ {
		index := SingleRouletteThrow(probabilities)
		if index < 0 || index >= len(probabilities) {
			t.Error("invalid segment index", index)
			return
		}
		// increment histogram to check distribution quality
		hist[index] += 1
	}
	t.Log(hist)
}
