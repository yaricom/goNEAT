package genetics

import (
	"testing"
	"math/rand"
	"sort"
	"math"
	"bytes"
	"encoding/gob"
)

// tests organisms sorting
func TestOrganisms(t *testing.T) {
	gnome := buildTestGenome(1)
	count := 100
	orgs := make(Organisms, count)
	var err error
	for i := 0; i < count; i++ {
		orgs[i], err = NewOrganism(rand.Float64(), gnome, 1)
		if err != nil {
			t.Error(err)
			return
		}
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
		orgs[i], err = NewOrganism(rand.Float64(), gnome, 1)
		if err != nil {
			t.Error(err)
			return
		}
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

func TestOrganism_MarshalBinary(t *testing.T) {
	gnome := buildTestGenome(1)
	org, err := NewOrganism(rand.Float64(), gnome, 1)
	if err != nil {
		t.Error(err)
		return
	}

	// Marshal to binary
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err = enc.Encode(org)
	if err != nil {
		t.Error(err)
		return
	}

	// Unmarshal and check if the same
	dec := gob.NewDecoder(&buf)
	dec_org := Organism{}
	err = dec.Decode(&dec_org)
	if err != nil {
		t.Error(err)
		return
	}

	// check results
	if org.Fitness != dec_org.Fitness {
		t.Error("org.Fitness != dec_org.Fitness")
	}

	dec_gnome := dec_org.Genotype
	if gnome.Id != dec_gnome.Id {
		t.Error("gnome.Id != dec_gnome.Id")
	}


	equals, err := gnome.IsEqual(dec_gnome)
	if !equals {
		t.Error(err)
	}
}
