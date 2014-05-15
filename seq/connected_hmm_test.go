package seq

import (
	"fmt"
	"hector/core"
	distr "hector/distribution"
	"math/rand"
	"testing"
)

func CompareSlice(x, y []VocIndex) bool {
	if len(x) != len(y) {
		return false
	}
	for n, vx := range x {
		if y[n] != vx {
			return false
		}
	}
	return true
}

func TestHMM(t *testing.T) {
	// config
	dim := 2
	numTest := 15

	// mixtures
	means1 := [][]float64{
		[]float64{-2, -2},
		[]float64{-2, -1},
	}
	priors1 := []float64{0.9, 0.1}
	means2 := [][]float64{
		[]float64{0, 0},
		[]float64{-1, -1},
	}
	priors2 := []float64{0.9, 0.1}
	means3 := [][]float64{
		[]float64{2, 2},
		[]float64{1, 2},
	}
	priors3 := []float64{0.6, 0.4}
	means4 := [][]float64{
		[]float64{2, 0},
		[]float64{1, 0},
	}
	priors4 := []float64{0.9, 0.1}
	means5 := [][]float64{
		[]float64{-1, -2},
		[]float64{-1, -1},
	}
	priors5 := []float64{0.9, 0.1}
	sigma := 0.5
	covs := [][]float64{
		[]float64{sigma * sigma, sigma * sigma},
		[]float64{sigma * sigma, sigma * sigma},
	}
	gmm1, _ := distr.NewGMM(dim, priors1, ToGMM(means1, covs))
	gmm2, _ := distr.NewGMM(dim, priors2, ToGMM(means2, covs))
	gmm3, _ := distr.NewGMM(dim, priors3, ToGMM(means3, covs))
	gmm4, _ := distr.NewGMM(dim, priors4, ToGMM(means4, covs))
	gmm5, _ := distr.NewGMM(dim, priors5, ToGMM(means5, covs))

	// model A
	transitionMatA := [][]float64{
		[]float64{0.8, 0.2, 0, 0},
		[]float64{0.0, 0.8, 0.2, 0},
		[]float64{0.0, 0.0, 0.8, 0.2},
	}
	gmmsA := []*distr.GMM{gmm1, gmm2, gmm3}
	var err error
	modelA, err := NewHMM(3,
		ToMat(transitionMatA),
		gmmsA,
	)
	if err != nil {
		t.Errorf("%s", err)
	}
	// model B
	transitionMatB := [][]float64{
		[]float64{0.8, 0.2, 0},
		[]float64{0, 0.8, 0.2},
	}
	gmmsB := []*distr.GMM{gmm5, gmm4}
	modelB, err := NewHMM(2,
		ToMat(transitionMatB),
		gmmsB,
	)
	if err != nil {
		t.Errorf("%s", err)
	}

	var models []*HMM = []*HMM{modelA, modelB}
	var cHMM = NewConnectedHMM(models)
	// generate data
	for cnt := 0; cnt < numTest; cnt++ {
		L := rand.Int()%10 + 2
		stdSeq := make([]VocIndex, L, L)
		// stdSeq := []VocIndex{1, 1, 0, 0, 1}
		// L := len(stdSeq)
		datas := make([]core.DenseVector, 0)
		for l := 0; l < L; l++ {
			v := VocIndex(rand.Int() % len(models))
			stdSeq[l] = v
			// v = stdSeq[l]
			datas = append(datas, GenerateData(dim, models[v])...)
		}
		bestResults := cHMM.ViterbiDecode(datas, L+L/2+2, 1)

		if len(bestResults) == 0 {
			t.Errorf("Fail to get reasonable result")
		}
		if !CompareSlice(stdSeq, bestResults[0].Seqs) {
			t.Errorf("DecodedSequence wrong std/decoded: %+v %+v", stdSeq, bestResults[0].Seqs)
		} else {
			fmt.Println("Succeed: Seq/score", stdSeq, bestResults[0].Score/float64(len(stdSeq)))
		}
	}
}
