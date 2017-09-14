package network

import (
	"github.com/yaricom/goNEAT/neat"
	"io"
	"fmt"
	"math/rand"
)

// TRAIT: A Trait is a group of parameters that can be expressed as a group more than one time. Traits save a genetic
// algorithm from having to search vast parameter landscapes on every node. Instead, each node can simply point to a trait
// and those traits can evolve on their own.
type Trait struct {
	// The trait ID
	TraitId int
	// The learned trait parameters
	Params []float64
}

func NewTrait() *Trait {
	trait := newTrait()
	return trait
}

func NewTraitWithParams(id int, p1, p2, p3, p4, p5, p6, p7, p8, p9 float64) *Trait {
	t := newTrait()
	t.TraitId = id
	t.Params[0] = p1
	t.Params[1] = p2
	t.Params[2] = p3
	t.Params[3] = p4
	t.Params[4] = p5
	t.Params[5] = p6
	t.Params[6] = p7
	t.Params[7] = 0
	return t
}

// The copy constructor
func NewTraitCopy(t *Trait) *Trait {
	nt := newTrait()
	nt.TraitId = t.TraitId
	for i, p := range t.Params {
		nt.Params[i] = p
	}
	return nt
}

// Special Constructor creates a new Trait which is the average of two existing traits passed in
func NewTraitAvrg(t1, t2 *Trait) *Trait {
	nt := newTrait()
	nt.TraitId = t1.TraitId
	for i := 0; i < neat.Num_trait_params; i++ {
		nt.Params[i] = (t1.Params[i] + t2.Params[i]) / 2.0
	}
	return nt
}

// The method to read Trait from input
func ReadTrait(r io.Reader) *Trait {
	nt := newTrait()
	fmt.Fscanf(r, "trait %d ", &nt.TraitId)
	for i := 0; i < neat.Num_trait_params; i++ {
		fmt.Scanf("%f ", &nt.Params[i])
	}
	return nt
}

// The default private constructor
func newTrait() *Trait {
	return &Trait{
		Params:make([]float64, neat.Num_trait_params),
	}
}

// The Trait methods
func (t *Trait) Mutate(trait_mutation_power, trait_param_mut_prob float64) {
	for i := 0; i < neat.Num_trait_params; i++ {
		if rand.Float64() > trait_param_mut_prob {
			t.Params[i] += float64(neat.RandPosNeg()) * rand.Float64() * trait_mutation_power
			if t.Params[i] < 0 { t.Params[i] = 0 }
		}
	}
}
func (t *Trait) WriteTrait(w io.Writer) {
	fmt.Fprintf(w, "trait %d ", t.TraitId)
	for _, p := range t.Params {
		fmt.Fprintf(w, "%f ", p)
	}
}

func (t *Trait) String() string {
	s := fmt.Sprintf("Trait # %d\t", t.TraitId)
	for _, p := range t.Params {
		s = fmt.Sprintf("%s %f", s, p)
	}
	return s
}
