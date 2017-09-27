package genetics

import (
	"testing"
	"github.com/yaricom/goNEAT/neat"
)

func TestNewPopulationRandom(t *testing.T) {
	size, in, out, nmax := 10, 3, 2, 5
	recurrent := false
	link_prob := 0.5
	conf := neat.Neat{
		CompatThreshold:0.5,
	}
	pop, err := NewPopulationRandom(size, in, out, nmax, recurrent, link_prob, &conf)
	if err != nil {
		t.Error(err)
	}
	if pop == nil {
		t.Error("pop == nil")
	}
	if len(pop.Organisms) != size {
		t.Error("len(pop.Organisms) != size")
	}
	if pop.currNodeId != 11 {
		t.Error("pop.currNodeId != 11")
	}
	if pop.currInnovNum != int64(101) {
		t.Error("pop.currInnovNum != 101")
	}
	if len(pop.Species) == 0 {
		t.Error("len(pop.Species) == 0")
	}
	t.Log(pop.Species)
}
