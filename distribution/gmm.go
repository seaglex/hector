package distribution

import (
	"errors"
	"math"
)

type GMM struct {
	NumMixture int
	Dimension  int
	Priors     []float64
	LogPriors  []float64
	Mixtures   []*DiagonalGaussian
}

func NewGMM(dim int, priors []float64, distribs []*DiagonalGaussian) (*GMM, error) {
	gmm := &GMM{
		NumMixture: len(distribs),
		Dimension:  dim,
	}
	err := gmm.UpdateModel(priors, distribs)
	if err != nil {
		return nil, err
	}
	return gmm, nil
}

// Comments:
// default GMM should always suceed, or please debug it!
func DefaultGMM(dim int) *GMM {
	distrib := DefaultDiagonalGaussian(dim)
	gmm, err := NewGMM(dim, []float64{1}, []*DiagonalGaussian{distrib})
	if err != nil {
		panic(err)
	}
	return gmm
}

func removeDistrib(priors []float64, distribs []*DiagonalGaussian) ([]float64, []*DiagonalGaussian, error) {
	var newPriors []float64 = make([]float64, 0)
	var newDistribs []*DiagonalGaussian = make([]*DiagonalGaussian, 0)
	for m := 0; m < len(priors); m++ {
		if priors[m] < 0 {
			return nil, nil, errors.New("GMM: Negative Prior")
		}
		if priors[m] > 0 {
			newPriors = append(newPriors, priors[m])
			newDistribs = append(newDistribs, distribs[m])
		}
	}
	return newPriors, newDistribs, nil
}

func (self *GMM) UpdateModel(priors []float64, distribs []*DiagonalGaussian) error {
	if len(priors) != len(distribs) {
		return errors.New("GMM: dimension mismatches in UpdateModel")
	}
	var err error
	priors, distribs, err = removeDistrib(priors, distribs)
	self.NumMixture = len(priors)
	if err != nil {
		return err
	}
	var logPriors []float64 = make([]float64, self.NumMixture, self.NumMixture)
	for m := 0; m < self.NumMixture; m++ {
		if priors[m] < 0 {
			return errors.New("GMM: Negative Prior")
		}
		if priors[m] > 0 {
			logPriors[m] = math.Log(priors[m])
		}
	}
	self.Priors = priors
	self.LogPriors = logPriors
	self.Mixtures = distribs
	return nil
}

func (self *GMM) SplitOne() error {
	if len(self.Mixtures) == 0 {
		return errors.New("The number of mixtures is 0")
	}
	var maxPrior float64 = 0
	var maxIndex int = 0
	for m := 0; m < self.NumMixture; m++ {
		if self.Priors[m] >= maxPrior {
			maxPrior = self.Priors[m]
			maxIndex = m
		}
	}
	var err error
	distrib1, distrib2, err := self.Mixtures[maxIndex].Split()
	if err != nil {
		return err
	}
	newprior := self.Priors[maxIndex] / 2
	newLogPrior := self.LogPriors[maxIndex] - math.Log(2)
	self.Priors[maxIndex] = newprior
	self.LogPriors[maxIndex] = newLogPrior
	self.Mixtures[maxIndex] = distrib1
	self.Priors = append(self.Priors, newprior)
	self.LogPriors = append(self.LogPriors, newLogPrior)
	self.Mixtures = append(self.Mixtures, distrib2)
	self.NumMixture = len(self.Mixtures)
	return nil
}

func (self *GMM) GetProbability(vec []float32) float64 {
	var pr float64 = 0
	for m := 0; m < self.NumMixture; m++ {
		logPr := self.Mixtures[m].GetLogProbability(vec)
		if self.Priors[m] > 0 {
			pr += math.Exp(logPr + self.LogPriors[m])
		}
	}
	return pr
}

func (self *GMM) GetLogProbabilities(vec []float32) []float64 {
	var logPrs = make([]float64, self.NumMixture, self.NumMixture)
	for m := 0; m < self.NumMixture; m++ {
		tmp := self.Mixtures[m].GetLogProbability(vec)
		logPrs[m] = self.LogPriors[m] + tmp
	}
	return logPrs
}

type GMMTrainer struct {
	MaxMixture      int
	NumMixture      int
	Dimension       int
	DistribTrainers []*DiagonalGaussianTrainer
	AccPriors       []float64
	DistribMixture  *GMM
}

func NewGMMTrainer(dim int, maxMixture int) *GMMTrainer {
	var defaultGMM *GMM = DefaultGMM(dim)
	var numMixture int = defaultGMM.NumMixture
	trainer := &GMMTrainer{
		MaxMixture:      maxMixture,
		NumMixture:      numMixture,
		Dimension:       dim,
		DistribTrainers: make([]*DiagonalGaussianTrainer, numMixture, numMixture),
		AccPriors:       make([]float64, numMixture, numMixture),
		DistribMixture:  defaultGMM,
	}
	trainer.initGMMTrainer(dim, numMixture)
	return trainer
}

func (self *GMMTrainer) initGMMTrainer(dim int, numMixture int) {
	self.NumMixture = numMixture
	self.Dimension = dim
	if self.AccPriors != nil {
		self.AccPriors = make([]float64, self.NumMixture, self.NumMixture)
		self.DistribTrainers = make([]*DiagonalGaussianTrainer, self.NumMixture, self.NumMixture)
		for m := 0; m < self.NumMixture; m++ {
			self.DistribTrainers[m] = NewDiagonalGaussianTrainer(self.Dimension)
		}
	}
	if len(self.AccPriors) < self.NumMixture {
		for n := 0; n < len(self.AccPriors); n++ {
			self.DistribTrainers[n].ResetTrainer(self.Dimension)
		}
		for n := len(self.AccPriors); n < self.NumMixture; n++ {
			self.AccPriors = append(self.AccPriors, 0)
			self.DistribTrainers = append(self.DistribTrainers, NewDiagonalGaussianTrainer(self.Dimension))
		}
	} else {
		// clear ACC prior
		for m := 0; m < self.NumMixture; m++ {
			self.AccPriors[m] = 0
		}
	}
}

func getMaxValue(prs []float64) float64 {
	var maxPr float64 = 0
	for _, pr := range prs {
		if pr > maxPr {
			maxPr = pr
		}
	}
	return maxPr
}

func (self *GMMTrainer) LearnCase(vec []float32, weight float64) float64 {
	var logPosteriors []float64
	logPosteriors = self.DistribMixture.GetLogProbabilities(vec)
	var maxPosterior float64 = getMaxValue(logPosteriors)
	var posteriors []float64 = make([]float64, self.NumMixture, self.NumMixture)
	var sumPr float64 = 0
	for m := 0; m < self.NumMixture; m++ {
		posteriors[m] = math.Exp(logPosteriors[m] - maxPosterior)
		sumPr += posteriors[m]
	}
	if self.NumMixture == 0 {
		return 0
	}
	weight *= 1.0 / sumPr
	for m := 0; m < self.NumMixture; m++ {
		wgt := weight * posteriors[m]
		self.AccPriors[m] += wgt
		self.DistribTrainers[m].LearnCase(vec, wgt)
	}
	return math.Log(sumPr) + maxPosterior
}

func (self *GMMTrainer) Optimize() error {
	var accWeight float64 = 0
	for m := 0; m < self.NumMixture; m++ {
		accWeight += self.AccPriors[m]
	}
	if accWeight <= 0 {
		return errors.New("GMMTrainer: sum of accumulated priors is non-positve")
	}
	var scale float64 = 1.0 / accWeight
	var priors = make([]float64, self.NumMixture, self.NumMixture)
	var distribs = make([]*DiagonalGaussian, self.NumMixture, self.NumMixture)
	for m := 0; m < self.NumMixture; m++ {
		priors[m] = scale * self.AccPriors[m]
		distrib, err := self.DistribTrainers[m].Distribution()
		if err != nil {
			return errors.New("GMMTrainer: fail to get gaussian distribution")
		}
		distribs[m] = distrib
	}
	self.DistribMixture.UpdateModel(priors, distribs)
	if self.NumMixture < self.MaxMixture {
		self.DistribMixture.SplitOne()
	}
	self.initGMMTrainer(self.DistribMixture.Dimension, self.DistribMixture.NumMixture)
	return nil
}

func (self *GMMTrainer) GMM() *GMM {
	result := *self.DistribMixture
	return &result
}
