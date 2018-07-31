package experiments

import (
	"testing"
	"bytes"
)

func TestExperiment_Write_Read(t *testing.T) {
	ex := Experiment{Id:1, Name:"Test Encode Decode", Trials:make(Trials, 3)}
	for i := 0; i < len(ex.Trials); i++ {
		ex.Trials[i] = *buildTestTrial(i + 1, 10)
	}

	// Write experiment
	var buff bytes.Buffer
	err := ex.Write(&buff)
	if err != nil {
		t.Error("Failed to write experiment")
	}

	// Read experiment
	data := buff.Bytes()
	new_ex := Experiment{}
	err = new_ex.Read(bytes.NewBuffer(data))
	if err != nil {
		t.Error("failed to read experiment")
	}

	// Deep compare results
	if ex.Id != new_ex.Id {
		t.Error("ex.Id != new_ex.Id")
	}

	if ex.Name != new_ex.Name {
		t.Error("ex.Name != new_ex.Name")
	}

	if len(ex.Trials) != len(new_ex.Trials) {
		t.Error("len(ex.Trials) != len(new_ex.Trials)")
		return
	}

	for i := 0; i < len(ex.Trials); i++ {
		deepCompareTrials(&ex.Trials[i], &new_ex.Trials[i], t)
	}
}
