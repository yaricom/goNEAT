package experiment

import (
	"bytes"
	"encoding/gob"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestTrial_Encode_Decode(t *testing.T) {
	trial := buildTestTrial(1, 3)

	var buff bytes.Buffer
	enc := gob.NewEncoder(&buff)

	// encode trial
	err := trial.Encode(enc)
	require.NoError(t, err, "failed to encode Trial")

	// decode trial
	data := buff.Bytes()
	dec := gob.NewDecoder(bytes.NewBuffer(data))

	decTrial := Trial{}
	err = decTrial.Decode(dec)
	require.NoError(t, err, "failed to decode trial")

	// do deep compare of Trail fields
	assert.EqualValues(t, *trial, decTrial)
}

func buildTestTrial(id, numGenerations int) *Trial {
	trial := Trial{Id: id, Generations: make([]Generation, numGenerations)}
	for i := 0; i < numGenerations; i++ {
		trial.Generations[i] = *buildTestGeneration(i+1, float64(i+1)*math.E)
	}
	return &trial
}
