package experiment

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestExperiment_Write_Read(t *testing.T) {
	ex := Experiment{Id: 1, Name: "Test Encode Decode", Trials: make(Trials, 3)}
	for i := 0; i < len(ex.Trials); i++ {
		ex.Trials[i] = *buildTestTrial(i+1, 10)
	}

	// Write experiment
	var buff bytes.Buffer
	err := ex.Write(&buff)
	require.NoError(t, err, "Failed to write experiment")

	// Read experiment
	data := buff.Bytes()
	newEx := Experiment{}
	err = newEx.Read(bytes.NewBuffer(data))
	require.NoError(t, err, "failed to read experiment")

	// Deep compare results
	assert.Equal(t, ex.Id, newEx.Id)
	assert.Equal(t, ex.Name, newEx.Name)
	require.Len(t, newEx.Trials, len(ex.Trials))

	for i := 0; i < len(ex.Trials); i++ {
		assert.EqualValues(t, ex.Trials[i], newEx.Trials[i])
	}
}
