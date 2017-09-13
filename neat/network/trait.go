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
type Trait interface {
	// Returns trait ID
	GetTraitId() int
	// Returns learned trait parameters
	GetParams() []float64

	// Perturb the trait parameters slightly
	Mutate(trait_mutation_power, trait_param_mut_prob float64)

	// Writes this Trait into given writer
	WriteTrait(w io.Writer)
}

func NewTrait() Trait {
	trait := newTrait()
	return &trait
}

func NewTraitWithParams(id int, p1, p2, p3, p4, p5, p6, p7, p8, p9 float64) Trait {
	t := newTrait()
	t.trait_id = id
	t.params[0] = p1
	t.params[1] = p2
	t.params[2] = p3
	t.params[3] = p4
	t.params[4] = p5
	t.params[5] = p6
	t.params[6] = p7
	t.params[7] = 0
	return &t
}

// The copy constructor
func NewTraitCopy(t Trait) Trait {
	nt := newTrait()
	nt.trait_id = t.GetTraitId()
	for i, p := range t.GetParams() {
		nt.params[i] = p
	}
	return &nt
}

// Special Constructor creates a new Trait which is the average of two existing traits passed in
func NewTraitAvrg(t1, t2 Trait) Trait {
	nt := newTrait()
	nt.trait_id = t1.GetTraitId()
	for i := 0; i < neat.Num_trait_params; i++ {
		nt.params[i] = (t1.GetParams()[i] + t2.GetParams()[i]) / 2.0
	}
	return &nt
}

// The method to read Trait from input
func ReadTrait(r *io.Reader) Trait {
	nt := newTrait()
	fmt.Scanf("%d ", &nt.trait_id)
	for i := 0; i < neat.Num_trait_params; i++ {
		fmt.Scanf("%f ", &nt.params[i])
	}
	return &nt
}


type trait struct {
	// Used in file saving and loading
	trait_id int
	// Keep traits in an array
	params []float64

}

// The default private constructor
func newTrait() trait {
	return trait{
		params:make([]float64, neat.Num_trait_params),
	}
}

// The Trait interface implementation
func (t *trait) GetTraitId() int {
	return t.trait_id
}
func (t *trait) GetParams() []float64 {
	return t.params
}
func (t *trait) Mutate(trait_mutation_power, trait_param_mut_prob float64) {
	for i := 0; i < neat.Num_trait_params; i++ {
		if rand.Float64() > trait_param_mut_prob {
			t.params[i] += float64(neat.RandPosNeg()) * rand.Float64() * trait_mutation_power
			if t.params[i] < 0 { t.params[i] = 0 }
		}
	}
}
func (t *trait) WriteTrait(w io.Writer) {
	fmt.Fprintf(w, "trait %d ", t.trait_id)
	for _, p := range t.params {
		fmt.Fprintf(w, "%f ", p)
	}
}

func (t *trait) String() string {
	s := fmt.Sprintf("Trait # %d\t", t.trait_id)
	for _, p := range t.params {
		s = fmt.Sprintf("%s %f", s, p)
	}
	return s
}
