package neat

import (
	"fmt"
	"math/rand"
	"errors"
	"github.com/yaricom/goNEAT/neat/utils"
)

// The number of parameters used in neurons that learn through habituation,
// sensitization, or Hebbian-type processes
const Num_trait_params = 8

var (
	ErrTraitsParametersCountMismatch = errors.New("traits parameters number mismatch")
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
	trait := newTrait(Num_trait_params)
	return trait
}

// The copy constructor
func NewTraitCopy(t *Trait) *Trait {
	nt := newTrait(len(t.Params))
	nt.Id = t.Id
	for i, p := range t.Params {
		nt.Params[i] = p
	}
	return nt
}

// Special Constructor creates a new Trait which is the average of two existing traits passed in
func NewTraitAvrg(t1, t2 *Trait) (*Trait, error) {
	if len(t1.Params) != len(t2.Params) {
		return nil, ErrTraitsParametersCountMismatch
	}
	nt := newTrait(len(t1.Params))
	nt.Id = t1.Id
	for i := 0; i < len(t1.Params); i++ {
		nt.Params[i] = (t1.Params[i] + t2.Params[i]) / 2.0
	}
	return nt, nil
}

// The default private constructor
func newTrait(lenght int) *Trait {
	return &Trait{
		Params:make([]float64, lenght),
	}
}

// Perturb the trait parameters slightly
func (t *Trait) Mutate(trait_mutation_power, trait_param_mut_prob float64) {
	for i := 0; i < len(t.Params); i++ {
		if rand.Float64() > trait_param_mut_prob {
			t.Params[i] += float64(utils.RandSign()) * rand.Float64() * trait_mutation_power
			if t.Params[i] < 0 {
				t.Params[i] = 0
			}
		}
	}
}

func (t *Trait) String() string {
	s := fmt.Sprintf("Trait #%d (", t.Id)
	for _, p := range t.Params {
		s = fmt.Sprintf("%s %f", s, p)
	}
	s = fmt.Sprintf("%s )", s)
	return s
}
