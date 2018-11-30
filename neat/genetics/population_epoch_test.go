package genetics

import (
	"testing"
	"github.com/yaricom/goNEAT/neat"
	"math/rand"
)

func runSequentialPopulationEpochExecutor_NextEpoch(pop *Population, conf *neat.NeatContext) error {
	ex := SequentialPopulationEpochExecutor{}

	for i := 0; i < 100; i++ {
		err := ex.NextEpoch(i + 1, pop, conf)
		if err != nil {
			return err
		}
	}
	return nil

}

func runParallelPopulationEpochExecutor_NextEpoch(pop *Population, conf *neat.NeatContext) error {
	ex := ParallelPopulationEpochExecutor{}

	for i := 0; i < 100; i++ {
		err := ex.NextEpoch(i + 1, pop, conf)
		if err != nil {
			return err
		}
	}
	return nil
}

func TestPopulationEpochExecutor_NextEpoch(t *testing.T) {
	rand.Seed(42)
	in, out, nmax, n := 3, 2, 15, 3
	recurrent := false
	link_prob := 0.8
	conf := neat.NeatContext{
		CompatThreshold:0.5,
		DropOffAge:1,
		PopSize: 30,
		BabiesStolen:10,
		RecurOnlyProb:0.2,
	}
	neat.LogLevel = neat.LogLevelInfo
	gen := newGenomeRand(1, in, out, n, nmax, recurrent, link_prob)
	pop, err := NewPopulation(gen, &conf)
	if err != nil {
		t.Error(err)
	}
	if pop == nil {
		t.Error("pop == nil")
	}

	// test sequential executor
	err = runSequentialPopulationEpochExecutor_NextEpoch(pop, &conf)
	if err != nil {
		t.Error(err)
	}

	// test parallel executor
	err = runParallelPopulationEpochExecutor_NextEpoch(pop, &conf)
	if err != nil {
		t.Error(err)
	}
}
