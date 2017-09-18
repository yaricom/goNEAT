package network

import (
	"testing"
	"fmt"
	"strings"
	"bytes"
)

// Tests Trait ReadTrait
func TestTrait_ReadTrait(t *testing.T)  {
	params := []float64 {
		0.40227575878298616, 0.0, 0.0, 0.0, 0.0, 0.3245553261200018, 0.0, 0.12248956525856575,
	}
	trait_id := 2
	trait_str := fmt.Sprintf("trait %d %g %g %g %g %g %g %g %g",
			trait_id, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
	trait := ReadTrait(strings.NewReader(trait_str))
	if trait.TraitId != trait_id {
		t.Error("trait.TraitId", trait_id, trait.TraitId)
	}
}

// Tests Trait WriteTrait
func TestTrait_WriteTrait(t *testing.T)  {
	params := []float64 {
		0.40227575878298616, 0.0, 0.0, 0.0, 0.0, 0.3245553261200018, 0.0, 0.12248956525856575,
	}
	trait_id := 2
	trait := NewTrait()
	trait.TraitId = trait_id
	trait.Params = params

	trait_str := fmt.Sprintf("trait %d %g %g %g %g %g %g %g %g ",
		trait_id, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])

	out_buffer := bytes.NewBufferString("")
	trait.WriteTrait(out_buffer)
	out_str := out_buffer.String()
	if trait_str != out_str {
		t.Errorf("Wrong trait serialization\n[%s]\n[%s]", trait_str, out_str)
	}
}
