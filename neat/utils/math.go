package utils

import (
	"math/rand"
)

// Returns subsequent random positive or negative integer value (1 or -1) to randomize value sign
func RandSign() int32 {
	v := rand.Int()
	if (v % 2) == 0 {
		return -1
	} else {
		return 1
	}
}
