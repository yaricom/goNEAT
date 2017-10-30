package experiments

import (
	"sort"
	"math"
)

// Floats provides descriptive statistics on a slice of float64 values
type Floats []float64

// Min returns the smallest value in the slice
func (x Floats) Min() float64 {
	if len(x) == 0 {
		return 0.0
	}
	sort.Float64s(x)
	return x[0]
}

// Max returns the greatest value in the slice
func (x Floats) Max() float64 {
	if len(x) == 0 {
		return 0.0
	}
	sort.Float64s(x)
	return x[len(x)-1]
}

// Sum returns the total of the values in the slice
func (x Floats) Sum() float64 {
	s := 0.0
	for _, v := range x {
		s += v
	}
	return s
}

// Mean returns the average of the values in the slice
func (x Floats) Mean() float64 {
	if len(x) == 0 {
		return 0.0
	}
	return x.Sum() / float64(len(x))
}

// Median returns the middle value in the slice
func (x Floats) Median() float64 {
	sort.Float64s(x)
	n := len(x)
	switch {
	case n == 0:
		return 0.0
	case n%2 == 0:
		return (x[n/2-1] + x[n/2]) / 2.0
	default:
		return x[n/2]
	}
}

func (x Floats) Q25() float64 {
	if len(x) == 0 {
		return 0.0
	}
	sort.Float64s(x)
	n := len(x) / 4
	return x[n]
}

func (x Floats) Q75() float64 {
	if len(x) == 0 {
		return 0.0
	}
	sort.Float64s(x)
	n := len(x) * 3 / 4
	return x[n]
}

// Variance returns the variance of the values in the slice
func (x Floats) Variance() float64 {
	if len(x) == 0 {
		return 0.0
	}
	m := x.Mean()
	s := 0.0
	for _, v := range x {
		s += (v - m) * (v - m)
	}
	return s
}

// Stdev returns the standard deviation of the values in the slice
func (x Floats) Stdev() float64 {
	if len(x) == 0 {
		return 0.0
	}
	v := x.Variance()
	return math.Sqrt(v / float64(len(x)))
}
