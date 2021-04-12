package neat

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestNewTraitAvrg(t *testing.T) {
	t1 := &Trait{Id: 1, Params: []float64{1, 2, 3, 4, 5, 6}}
	t2 := &Trait{Id: 2, Params: []float64{2, 3, 4, 5, 6, 7}}

	tr, err := NewTraitAvrg(t1, t2)
	require.NoError(t, err, "failed to create trait")
	assert.Equal(t, t1.Id, tr.Id, "wrong trait ID")

	for i, p := range tr.Params {
		expected := (t1.Params[i] + t2.Params[i]) / 2.0
		assert.Equal(t, expected, p, "wrong parameter at: %d", i)
	}
}

func TestNewTraitCopy(t *testing.T) {
	t1 := &Trait{Id: 1, Params: []float64{1, 2, 3, 4, 5, 6}}

	tr := NewTraitCopy(t1)
	assert.Equal(t, t1.Id, tr.Id, "wrong trait ID")
	for i, p := range tr.Params {
		assert.Equal(t, t1.Params[i], p, "Wrong parameter at: %d", i)
	}
}
