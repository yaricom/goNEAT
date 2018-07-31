package experiments

import (
	"testing"
	"math"
	"bytes"
	"encoding/gob"
)

func TestTrial_Encode_Decode(t *testing.T) {
	trial := buildTestTrial(1, 3)

	var buff bytes.Buffer
	enc := gob.NewEncoder(&buff)

	// encode trial
	err := trial.Encode(enc)
	if err != nil {
		t.Error("failed to encode Trial", err)
		return
	}

	// decode trial
	data := buff.Bytes()
	dec := gob.NewDecoder(bytes.NewBuffer(data))

	dec_trial := Trial{}
	err = dec_trial.Decode(dec)
	if err != nil {
		t.Error("failed to decode trial", err)
		return
	}

	// do deep compare of Trail fields
	deepCompareTrials(trial, &dec_trial, t)
}

func deepCompareTrials(first, second *Trial, t *testing.T) {
	if first.Id != second.Id {
		t.Error("first.Id != second.Id")
	}
	if len(first.Generations) != len(second.Generations) {
		t.Error("len(first.Generations) != len(second.Generations)")
		return
	}

	for i := 0; i < len(first.Generations); i++ {
		deepCompareGenerations(&first.Generations[i], &second.Generations[i], t)
	}
}

func buildTestTrial(id, num_generations int) *Trial {
	trial := Trial{Id:id, Generations:make([]Generation, num_generations)}
	for i := 0; i < num_generations; i++ {
		trial.Generations[i] = *buildTestGeneration(i + 1, float64(i + 1) * math.E)
	}
	return &trial
}