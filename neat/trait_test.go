package neat

import (
	"testing"
)

func TestNewTraitAvrg(t *testing.T) {
	t1 := &Trait{Id:1, Params:[]float64{1, 2, 3, 4, 5, 6}}
	t2 := &Trait{Id:2, Params:[]float64{2, 3, 4, 5, 6, 7}}

	tr, err := NewTraitAvrg(t1, t2)
	if err != nil {
		t.Error(err)
	}
	if tr.Id != t1.Id {
		t.Error("tr.Id != t1.Id", tr.Id)
	}
	for i, p := range tr.Params {
		if p != (t1.Params[i] + t2.Params[i]) / 2.0 {
			t.Error("Wrong parameter at: ", i)
		}
	}
}

func TestNewTraitCopy(t *testing.T) {
	t1 := &Trait{Id:1, Params:[]float64{1, 2, 3, 4, 5, 6}}

	tr := NewTraitCopy(t1)
	if tr.Id != t1.Id {
		t.Error("tr.Id != t1.Id", tr.Id)
	}
	for i, p := range tr.Params {
		if p != t1.Params[i] {
			t.Error("Wrong parameter at: ", i)
		}
	}
}
