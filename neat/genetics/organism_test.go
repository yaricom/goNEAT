package genetics

import (
	"testing"
	"math/rand"
	"sort"
	"math"
)

// tests organisms sorting
func TestOrganisms(t *testing.T) {
	gnome := buildTestGenome(1)
	count := 100
	orgs := make(Organisms, count)
	for i := 0; i < count; i++ {
		orgs[i] = NewOrganism(rand.Float64(), gnome, 1)
	}

	// sort ascending
	sort.Sort(orgs)
	fit := 0.0
	for _, o := range orgs {
		if o.Fitness < fit {
			t.Error("Wrong ascending sort order")
		}
		fit = o.Fitness
	}

	// sort descending
	for i := 0; i < count; i++ {
		orgs[i] = NewOrganism(rand.Float64(), gnome, 1)
	}
	sort.Sort(sort.Reverse(orgs))
	fit = math.MaxFloat64
	for _, o := range orgs {
		if o.Fitness > fit {
			t.Error("Wrong ascending sort order")
		}
		fit = o.Fitness
	}
}
