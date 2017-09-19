package neat

import (
	"io"
	"fmt"
	"math/rand"
)

// TRAIT: A Trait is a group of parameters that can be expressed as a group more than one time. Traits save a genetic
// algorithm from having to search vast parameter landscapes on every node. Instead, each node can simply point to a trait
// and those traits can evolve on their own.
type Trait struct {
	// The trait ID
	Id     int
	// The learned trait parameters
	Params []float64
}

func NewTrait() *Trait {
	trait := newTrait()
	return trait
}

// The copy constructor
func NewTraitCopy(t *Trait) *Trait {
	nt := newTrait()
	nt.Id = t.Id
	for i, p := range t.Params {
		nt.Params[i] = p
	}
	return nt
}

// Special Constructor creates a new Trait which is the average of two existing traits passed in
func NewTraitAvrg(t1, t2 *Trait) *Trait {
	nt := newTrait()
	nt.Id = t1.Id
	for i := 0; i < Num_trait_params; i++ {
		nt.Params[i] = (t1.Params[i] + t2.Params[i]) / 2.0
	}
	return nt
}

// The method to read Trait from input
func ReadTrait(r io.Reader) *Trait {
	nt := newTrait()
	fmt.Fscanf(r, "%d ", &nt.Id)
	for i := 0; i < Num_trait_params; i++ {
		fmt.Fscanf(r, "%g ", &nt.Params[i])
	}
	return nt
}

// The default private constructor
func newTrait() *Trait {
	return &Trait{
		Params:make([]float64, Num_trait_params),
	}
}

// Perturb the trait parameters slightly
func (t *Trait) Mutate(trait_mutation_power, trait_param_mut_prob float64) {
	for i := 0; i < Num_trait_params; i++ {
		if rand.Float64() > trait_param_mut_prob {
			t.Params[i] += float64(RandPosNeg()) * rand.Float64() * trait_mutation_power
			if t.Params[i] < 0 { t.Params[i] = 0 }
		}
	}
}
// Dump trait to a writer
func (t *Trait) WriteTrait(w io.Writer) {
	fmt.Fprintf(w, "%d ", t.Id)
	for _, p := range t.Params {
		fmt.Fprintf(w, "%g ", p)
	}
}

func (t *Trait) String() string {
	s := fmt.Sprintf("Trait # %d\t", t.Id)
	for _, p := range t.Params {
		s = fmt.Sprintf("%s %f", s, p)
	}
	return s
}
