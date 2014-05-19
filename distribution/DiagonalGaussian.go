package distribution

import (
	"errors"
	"fmt"
	"math"
)

type DiagonalGaussian struct {
	Means      []float64
	Covs       []float64
	Precisions []float64
	LogC       float64
}

const (
	DEFAULT_VARIANCE = 1
	INIT_COUNT       = 1
)

func NewDiagonalGaussian(means []float64, covs []float64) (*DiagonalGaussian, error) {
	gaussian := &DiagonalGaussian{}
	err := gaussian.UpdateModel(means, covs)
	if err != nil {
		return nil, err
	}
	return gaussian, nil
}

// Comments:
// default GMM should always suceed, or please debug it!
func DefaultDiagonalGaussian(dim int) *DiagonalGaussian {
	var means []float64 = make([]float64, dim, dim)
	var covs []float64 = make([]float64, dim, dim)
	for n := 0; n < dim; n++ {
		covs[n] = DEFAULT_VARIANCE
	}
	gaussin, err := NewDiagonalGaussian(means, covs)
	if err != nil {
		panic(err)
	}
	return gaussin
}

func (self *DiagonalGaussian) Split() (*DiagonalGaussian, *DiagonalGaussian, error) {
	var means1 []float64 = make([]float64, self.GetDimension(), self.GetDimension())
	var means2 []float64 = make([]float64, self.GetDimension(), self.GetDimension())
	for d := 0; d < self.GetDimension(); d++ {
		stdev := math.Sqrt(self.Covs[d])
		means1[d] = self.Means[d] + stdev
		means2[d] = self.Means[d] - stdev
	}
	var gaussian1 DiagonalGaussian = *self
	gaussian1.Means = means1
	var gaussian2 DiagonalGaussian = *self
	gaussian2.Means = means2
	return &gaussian1, &gaussian2, nil
}

func (gaussian *DiagonalGaussian) UpdateModel(means []float64, covs []float64) error {
	if len(means) != len(covs) {
		return errors.New("DiagonalGaussian: dimensions of Means/covs mismatch")
	}
	gaussian.Means = means
	gaussian.Covs = covs
	var logC float64 = 0
	var precisions = make([]float64, len(means), len(means))
	for n, cov := range covs {
		if cov < 0 {
			return errors.New("DiagonalGaussian: the cov is negative")
		} else if cov == 0 {
			continue
		}
		precisions[n] = 1.0 / cov
		logC += math.Log(cov)
	}
	logC += math.Log(2*math.Pi) * float64(len(covs))
	logC = -0.5 * logC
	gaussian.LogC = logC
	gaussian.Precisions = precisions
	return nil
}

func (self *DiagonalGaussian) GetDimension() int {
	return len(self.Means)
}

// Comments:
// The function will never return error except dimension mismatch
// It's caller' duty to ensure this
func (self *DiagonalGaussian) GetLogProbability(vec []float32) float64 {
	if len(vec) != self.GetDimension() {
		panic(errors.New("DiagonalGaussian: input dimension incorrect"))
	}
	var dist float64 = 0
	for n, x := range vec {
		if self.Covs[n] == 0.0 {
			if self.Means[n] != float64(x) {
				return -math.MaxFloat64
			}
			continue
		}
		d := float64(x) - self.Means[n]
		dist += d * d * self.Precisions[n]
	}
	return -0.5*dist + self.LogC
}

type DiagonalGaussianTrainer struct {
	Dimension int
	AccMeans  []float64
	AccCovs   []float64
	AccWeight float64
}

func NewDiagonalGaussianTrainer(dim int) *DiagonalGaussianTrainer {
	var trainer *DiagonalGaussianTrainer = &DiagonalGaussianTrainer{}
	trainer.ResetTrainer(dim)
	return trainer
}

func (self *DiagonalGaussianTrainer) ResetTrainer(dim int) {
	self.Dimension = dim
	if self.AccMeans != nil || len(self.AccMeans) != dim {
		self.AccMeans = make([]float64, dim, dim)
		self.AccCovs = make([]float64, dim, dim)
	}
	self.AccWeight = INIT_COUNT
	for n := 0; n < dim; n++ {
		self.AccMeans[n] = 0
		self.AccCovs[n] = DEFAULT_VARIANCE * INIT_COUNT
	}
	return
}

// Comments:
// No dangerous operation, no error expected
// If panic(), it's the duty of caller
func (self *DiagonalGaussianTrainer) LearnCase(vec []float32, weight float64) {
	if len(vec) != self.Dimension {
		panic(errors.New("DiagonalGaussianTrainer: dimension incorrect"))
	}
	self.AccWeight += weight
	for n, x := range vec {
		self.AccMeans[n] += float64(x) * weight
		self.AccCovs[n] += float64(x) * float64(x) * weight
	}
}

// Return: a standalone DiagonalGaussian (which will not been updated)
func (self *DiagonalGaussianTrainer) Distribution() (*DiagonalGaussian, error) {
	if self.AccWeight <= 0 {
		return nil, errors.New("DiagonalGaussianTrainer: The accumlated weights is not positive")
	}
	var means = make([]float64, self.Dimension, self.Dimension)
	var covs = make([]float64, self.Dimension, self.Dimension)
	scalar := 1.0 / self.AccWeight
	minCov := DEFAULT_VARIANCE * INIT_COUNT * scalar
	for n := 0; n < self.Dimension; n++ {
		means[n] = scalar * self.AccMeans[n]
		covs[n] = math.Max(scalar*self.AccCovs[n]-means[n]*means[n], minCov)
		if covs[n] <= 0 {
			panic(fmt.Sprint("DiagonalGaussianTrainer: covs[", n, "] =", covs[n]))
		}
	}
	return NewDiagonalGaussian(means, covs)
}
