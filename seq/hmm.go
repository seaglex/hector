package seq

import (
	"errors"
	"fmt"
	"hector/core"
	distr "hector/distribution"
	"math"
	"strings"
)

// HMM:
// To make things simple, it assume:
// The first point always be state 0, and the virtual-end point is always be state numState
// Add a fake state 0 if necessary
type HMM struct {
	NumState int
	PI       core.DenseVector64
	A        *core.Matrix
	B        []*distr.GMM
}

func CheckStateTransition(numState int, stateTransition *core.Matrix) error {
	if stateTransition == nil {
		return errors.New("HMM: nil argument")
	}
	var prOut float64 = 0
	for src, vec := range stateTransition.Data {
		if src < 0 {
			errors.New("HMM: source state is negative")
		} else if int(src) >= numState {
			return errors.New("HMM: source state >= numState (numState is end state)")
		}
		for dst, pr := range vec.Data {
			if pr < 0 {
				return errors.New("HMM: negative transition probability")
			}
			if dst < 0 {
				errors.New("HMM: destination state is negative")
			} else if int(dst) > numState {
				errors.New("HMM: dstination state > numState")
			} else if int(dst) == numState {
				prOut += pr
			}
		}
	}
	if prOut <= 0 {
		return errors.New("HMM: No chance to go to end state (numState)")
	}
	return nil
}

func (self *HMM) String() string {
	lines := []string{
		fmt.Sprint("HMM:", self.NumState),
		core.Mat2String(self.A),
	}
	for _, gmm := range self.B {
		lines = append(lines, gmm.String())
	}
	return strings.Join(lines, "\n")
}
func NewHMM(numState int, stateTransition *core.Matrix, stateObservation []*distr.GMM) (*HMM, error) {
	if numState <= 1 {
		return nil, errors.New("HMM: numState <= 1")
	}
	if stateTransition == nil || stateObservation == nil {
		return nil, errors.New("HMM: Nil Arguments")
	}
	err := CheckStateTransition(numState, stateTransition)
	if err != nil {
		return nil, err
	}
	if numState != len(stateObservation) {
		return nil, errors.New("HMM: NumState mismatchs between Pi and B")
	}
	var pi core.DenseVector64 = make(core.DenseVector64, numState+1, numState+1)
	pi[0] = 1
	hmm := &HMM{
		NumState: numState,
		PI:       pi,
		A:        stateTransition,
		B:        stateObservation,
	}
	return hmm, nil
}

func (self *HMM) GetStateProbabilities(x []float32) core.DenseVector64 {
	var prs []float64 = make([]float64, self.NumState+1, self.NumState+1)
	for n, model := range self.B {
		prs[n] = model.GetProbability(x)
	}
	return prs
}

func (self *HMM) Decode(seq []core.DenseVector) (alphas []core.DenseVector64, betas []core.DenseVector64, gammas []core.DenseVector64, prs []core.DenseVector64, score float64) {
	var T int = len(seq)
	if T == 0 {
		panic(errors.New("HMM: sequence length = 0"))
	}

	alphas = make([]core.DenseVector64, T, T)
	betas = make([]core.DenseVector64, T, T)
	gammas = make([]core.DenseVector64, T, T)
	prs = make([]core.DenseVector64, T, T)
	var scale = make([]float64, T, T)

	prs[0] = self.GetStateProbabilities(seq[0])
	alphas[0] = core.MultiplyElemWise(self.PI, prs[0])
	scale[0] = 1.0 / core.Sum(alphas[0])
	alphas[0].Scale(scale[0])
	for t := 1; t < T; t++ {
		prs[t] = self.GetStateProbabilities(seq[t])
		alphas[t] = core.MultiplyVecSparseMat(alphas[t-1], self.A)
		alphas[t].MultiplyElemWise(prs[t])
		scale[t] = 1.0 / core.Sum(alphas[t])
		alphas[t].Scale(scale[t])
	}
	betas[T-1] = core.GetSparseMatColumn(self.A, self.NumState+1, self.NumState)
	var prEnd float64 = 0
	for s := 0; s < self.NumState; s++ {
		prEnd += alphas[T-1][s] * betas[T-1][s]
	}
	betas[T-1].Scale(scale[T-1])
	for t := T - 2; t >= 0; t-- {
		betas[t] = core.MultiplySparseMatVec(self.A, core.MultiplyElemWise(prs[t+1], betas[t+1]))
		betas[t].Scale(scale[t])
	}
	// fmt.Println(prEnd)
	for t := 0; t < T; t++ {
		gammas[t] = core.MultiplyElemWise(alphas[t], betas[t])
		total := core.Sum(gammas[t])
		if total <= 0 {
			panic("HMM: sum of gamma = 0")
		}
		// total / scale[t] == prEnd
		// fmt.Print(total/scale[t], " ")
		gammas[t].Scale(1.0 / total)
	}
	// fmt.Println()
	score = math.Log(prEnd)
	for _, s := range scale {
		score -= math.Log(s)
	}
	return alphas, betas, gammas, prs, score
}

type HMMTrainer struct {
	NumState            int
	InitTransitionCount *core.Matrix
	Model               *HMM
	AccTransitionCount  *core.Matrix
	GMMTrainers         []*distr.GMMTrainer
}

func NewHMMTrainer(dim int, numMixture int,
	numState int, initTransitionCount *core.Matrix) (*HMMTrainer, error) {
	trainer := &HMMTrainer{
		NumState:            numState,
		InitTransitionCount: initTransitionCount,
	}
	var err error
	var gmms []*distr.GMM = make([]*distr.GMM, numState, numState)
	var gmmTrainers []*distr.GMMTrainer = make([]*distr.GMMTrainer, numState, numState)
	for n := 0; n < numState; n++ {
		gmms[n] = distr.DefaultGMM(dim)
		gmmTrainers[n] = distr.NewGMMTrainer(dim, numMixture)
	}
	trainer.Model, err = NewHMM(numState, trainer.Normalize(initTransitionCount), gmms)
	if err != nil {
		return nil, err
	}
	trainer.AccTransitionCount = core.CloneMatrix(trainer.InitTransitionCount)
	trainer.GMMTrainers = gmmTrainers
	return trainer, nil
}

func (self *HMMTrainer) LearnCase(seq []core.DenseVector) float64 {
	alphas, betas, gammas, prs, score := self.Model.Decode(seq)
	var T int = len(seq)
	for s, vec := range self.Model.A.Data {
		accTranCount, _ := self.AccTransitionCount.Data[s]
		for r, transitionPr := range vec.Data {
			var tmp float64 = 0
			if int(r) != self.NumState {
				for t := 0; t < T-1; t++ {
					tmp += alphas[t][s] * transitionPr * prs[t+1][r] * betas[t+1][r]
				}
			} else {
				tmp = alphas[T-1][s] * transitionPr
			}
			accTranCount.Data[r] += tmp
		}
	}
	// fmt.Println(alphas, "\n")
	// fmt.Println(betas, "\n")
	// fmt.Println(gammas, "\n")
	for t := 0; t < T; t++ {
		o := seq[t]
		gamma := gammas[t]
		for s := 0; s < self.NumState; s++ {
			self.GMMTrainers[s].LearnCase(o, gamma[s])
		}
	}
	return score
}

func (self *HMMTrainer) Normalize(mat *core.Matrix) *core.Matrix {
	var transitionMat *core.Matrix = core.NewMatrix()
	for s, vec := range mat.Data {
		var sum float64 = 0
		for _, val := range vec.Data {
			sum += val
		}
		if sum <= 0 {
			continue
		}
		scale := 1.0 / sum
		for r, val := range vec.Data {
			transitionMat.SetValue(int64(s), int64(r), val*scale)
		}
	}
	return transitionMat
}

func (self *HMMTrainer) Optimize() error {
	var gmms []*distr.GMM = make([]*distr.GMM, self.NumState, self.NumState)
	for n := 0; n < self.NumState; n++ {
		self.GMMTrainers[n].Optimize()
		gmms[n] = self.GMMTrainers[n].GMM()
	}
	var err error
	transitionMat := self.Normalize(self.AccTransitionCount)
	self.Model, err = NewHMM(self.NumState, transitionMat, gmms)
	if err != nil {
		return err
	}
	self.AccTransitionCount = core.CloneMatrix(self.InitTransitionCount)
	return nil
}

func (self *HMMTrainer) HMM() *HMM {
	model := (*self.Model)
	return &model
}
