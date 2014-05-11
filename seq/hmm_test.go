package seq

import (
	"fmt"
	"hector/core"
	distr "hector/distribution"
	"math"
	"math/rand"
	"testing"
)

func ToMat(raw [][]float64) *core.Matrix {
	mat := core.NewMatrix()
	for r, vec := range raw {
		for c, val := range vec {
			mat.SetValue(int64(r), int64(c), val)
		}
	}
	return mat
}

func ToGMM(means [][]float64, covs [][]float64) []*distr.DiagonalGaussian {
	gaussians := make([]*distr.DiagonalGaussian, 0)
	for n := 0; n < len(means); n++ {
		g, _ := distr.NewDiagonalGaussian(means[n], covs[n])
		gaussians = append(gaussians, g)
	}
	return gaussians
}

func Sample(vec []float64) int {
	x := rand.Float64()
	var acc float64 = 0
	for n, val := range vec {
		acc += val
		if x < acc {
			return int(n)
		}
	}
	return len(vec) - 1
}

func SampleVector(vec *core.Vector) int {
	x := rand.Float64()
	var acc float64 = 0
	var lastN int64 = -1
	for n, val := range vec.Data {
		acc += val
		if x < acc {
			return int(n)
		}
		lastN = n
	}
	return int(lastN)
}

func TestHMM(t *testing.T) {
	// config
	numState := 3
	dim := 2
	numMixture := 2
	numItr := 15
	numData := 1000

	transitionMat := [][]float64{
		[]float64{0.8, 0.2, 0, 0},
		[]float64{0.0, 0.8, 0.2, 0},
		[]float64{0.0, 0.0, 0.8, 0.2},
	}
	means1 := [][]float64{
		[]float64{-2, -2},
		[]float64{-2, -1},
	}
	priors1 := []float64{0.9, 0.1}
	means2 := [][]float64{
		[]float64{-1, -1},
		[]float64{0, 0},
	}
	priors2 := []float64{0.9, 0.1}
	means3 := [][]float64{
		[]float64{2, 2},
		[]float64{1, 2},
	}
	priors3 := []float64{0.6, 0.4}
	covs := [][]float64{
		[]float64{0.5, 0.5},
		[]float64{0.5, 0.5},
	}

	initMat := [][]float64{
		[]float64{0.5, 0.5, 0, 0},
		[]float64{0, 0.5, 0.5, 0},
		[]float64{0, 0, 0.5, 0.5},
	}
	gmms := []*distr.GMM{}
	gmm, _ := distr.NewGMM(dim, priors1, ToGMM(means1, covs))
	gmms = append(gmms, gmm)
	gmm, _ = distr.NewGMM(dim, priors2, ToGMM(means2, covs))
	gmms = append(gmms, gmm)
	gmm, _ = distr.NewGMM(dim, priors3, ToGMM(means3, covs))
	gmms = append(gmms, gmm)
	var err error
	stdHMM, err := NewHMM(numState,
		ToMat(transitionMat),
		gmms,
	)
	if err != nil {
		t.Errorf("%s", err)
	}
	hmmTrainer, err := NewHMMTrainer(dim, numMixture, numState, ToMat(initMat))
	if err != nil {
		t.Errorf("%s", err)
	}

	// generate data
	datas := make([][]core.DenseVector, 1, numData)
	datas[0] = []core.DenseVector{
		core.DenseVector{-2, -2},
		core.DenseVector{-1, -1},
		core.DenseVector{2, 2},
	}
	for n := 1; n < numData; n++ {
		data := []core.DenseVector{}
		s := Sample(stdHMM.PI)
		for true {
			if s == stdHMM.NumState {
				break
			}
			gmm := stdHMM.B[s]
			m := Sample(gmm.Priors)
			vec := make(core.DenseVector, dim, dim)
			for d := 0; d < dim; d++ {
				vec[d] = float32(rand.NormFloat64()*gmm.Mixtures[m].Covs[d] + gmm.Mixtures[m].Means[d])
			}
			data = append(data, vec)
			s = SampleVector(stdHMM.A.GetRow(int64(s)))
		}
		datas = append(datas, data)
	}

	// test
	var stdScore float64 = 0
	for n := 0; n < numData; n++ {
		_, _, _, _, score := stdHMM.Decode(datas[n])
		stdScore += score
	}
	stdScore /= float64(numData)

	var trainScore float64 = 0.0
	for itr := 0; itr < numItr; itr++ {
		trainScore = 0
		for _, data := range datas {
			trainScore += hmmTrainer.LearnCase(data)
		}
		hmmTrainer.Optimize()
		trainScore /= float64(numData)
		fmt.Println("itr/score:", itr, trainScore)
		// fmt.Println(hmmTrainer.HMM().String())
	}
	resultHMM := hmmTrainer.HMM()
	fmt.Println(resultHMM.String())
	var resultScore float64 = 0
	for n := 0; n < numData; n++ {
		_, _, _, _, score := resultHMM.Decode(datas[n])
		resultScore += score
	}
	resultScore /= float64(numData)

	if (stdScore-resultScore)/math.Abs(stdScore) > 0.05 {
		t.Errorf("The training result is far from optima. Optimal/Train/Result: %f %f %f", stdScore, trainScore, resultScore)
	} else {
		fmt.Println("Optimal/Train/Result:", stdScore, trainScore, resultScore)
	}
}
