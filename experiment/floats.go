package experiment

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
	"math"
)

// Floats provides descriptive statistics on a slice of float64 values
type Floats []float64

// Min returns the smallest value in the slice
func (x Floats) Min() float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return floats.Min(x)
}

// Max returns the greatest value in the slice
func (x Floats) Max() float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return floats.Max(x)
}

// Sum returns the total of the values in the slice
func (x Floats) Sum() float64 {
	return floats.Sum(x)
}

// Mean returns the average of the values in the slice
func (x Floats) Mean() float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return stat.Mean(x, nil)
}

// MeanVariance returns the sample mean and unbiased variance of the values in the slice
func (x Floats) MeanVariance() []float64 {
	if len(x) == 0 {
		return []float64{math.NaN(), math.NaN()}
	}
	m, v := stat.MeanVariance(x, nil)
	return []float64{m, v}
}

// Median returns the middle value in the slice (50% quantile)
func (x Floats) Median() float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return stat.Quantile(0.5, stat.Empirical, x, nil)
}

// Q25 is the 25% quantile
func (x Floats) Q25() float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return stat.Quantile(0.25, stat.Empirical, x, nil)
}

// Q75 is the 75% quantile
func (x Floats) Q75() float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return stat.Quantile(0.75, stat.Empirical, x, nil)
}

// Variance returns the variance of the values in the slice
func (x Floats) Variance() float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return stat.Variance(x, nil)
}

// StdDev returns the standard deviation of the values in the slice
func (x Floats) StdDev() float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return stat.StdDev(x, nil)
}
