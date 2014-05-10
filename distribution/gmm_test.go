package distribution

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestGMM(t *testing.T) {
	var numIter = 30
	var numData = 10000
	var means [][2]float64 = make([][2]float64, 0, 0)
	var covs [][2]float64 = make([][2]float64, 0, 0)
	var accPriors []float64 = make([]float64, 0)
	// -3, 3
	means = append(means, [2]float64{-3, 3})
	covs = append(covs, [2]float64{0.25, 1})
	accPriors = append(accPriors, 0.2)
	// -1, 1
	means = append(means, [2]float64{-1, 1})
	covs = append(covs, [2]float64{0.25, 1})
	accPriors = append(accPriors, 0.4)
	// 1, -1
	means = append(means, [2]float64{1, -1})
	covs = append(covs, [2]float64{1, 0.25})
	accPriors = append(accPriors, 0.7)
	// 3, -3
	means = append(means, [2]float64{3, -3})
	covs = append(covs, [2]float64{1, 1})
	accPriors = append(accPriors, 1)

	// generate data
	var gaussians = make([]*DiagonalGaussian, len(means), len(means))
	var priors = make([]float64, len(means), len(means))
	datas := make([][2]float32, 0)
	n := 0
	for m := 0; m < len(means); m++ {
		mean := means[m]
		cov := covs[m]
		for ; n < int(float64(numData)*accPriors[m]); n++ {
			data := [2]float32{}
			for d := 0; d < 2; d++ {
				data[d] = float32(mean[d] + rand.NormFloat64()*math.Sqrt(float64(cov[d])))
			}
			datas = append(datas, data)
		}
		gaussians[m], _ = NewDiagonalGaussian(mean[:], cov[:])
		if m == 0 {
			priors[m] = accPriors[m]
		} else {
			priors[m] = accPriors[m] - accPriors[m-1]
		}
	}
	var stdGMM, _ = NewGMM(2, priors, gaussians)
	var optimalScore float64 = 0
	for _, data := range datas {
		optimalScore += math.Log(stdGMM.GetProbability(data[:]))
	}
	optimalScore /= float64(len(datas))

	gmmTrainer := NewGMMTrainer(2, 4)
	var score float64 = 0
	for itr := 0; itr < numIter; itr++ {
		score = 0
		for _, data := range datas {
			score += gmmTrainer.LearnCase(data[:], 1)
		}
		gmmTrainer.Optimize()
		score /= float64(len(datas))
		fmt.Println("itr/score", itr, score)
	}
	fmt.Println("Optimal/GMM", optimalScore, score)
	gmm := gmmTrainer.GMM()
	fmt.Println("mixture", gmm.NumMixture)
	for m := 0; m < gmm.NumMixture; m++ {
		fmt.Println(gmm.Priors[m])
		fmt.Println(gmm.Mixtures[m])
	}

	if math.Abs((optimalScore-score)/optimalScore) > 0.05 {
		t.Errorf("The difference between optimal and trained is too large: %f %f", optimalScore, score)
	}
}
