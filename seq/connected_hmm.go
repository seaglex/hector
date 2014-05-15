package seq

import (
	"container/heap"
	"fmt"
	"hector/core"
)

type ConnectedHMM struct {
	HMMs      []*HMM
	LangModel *core.Matrix
}

func NewConnectedHMM(hmms []*HMM) *ConnectedHMM {
	return &ConnectedHMM{HMMs: hmms}
}

// auxilliary - Decoded Sequence
type ModelTrack struct {
	LastV      VocIndex
	LastStartT int
}

// auxilliary - Decoded Sequence
type DecodedSequence struct {
	Score  float64
	Seqs   []VocIndex
	Tracks []*ModelTrack
}

// auxilliary - the end states
type ScoreEndStateItem struct {
	Score float64
	Level LevelIndex
	Model VocIndex
	State StateIndex
}
type ScoreEndStatePQueue []*ScoreEndStateItem

func (x *ScoreEndStatePQueue) Len() int {
	return len(*x)
}

func (x *ScoreEndStatePQueue) Swap(i, j int) {
	tmp := (*x)[i]
	(*x)[i] = (*x)[j]
	(*x)[j] = tmp
}

// inverse order: higher score, higher order
func (x *ScoreEndStatePQueue) Less(i, j int) bool {
	return (*x)[i].Score > (*x)[j].Score
}

func (x *ScoreEndStatePQueue) Push(item interface{}) {
	*x = append(*x, item.(*ScoreEndStateItem))
}

func (x *ScoreEndStatePQueue) Pop() interface{} {
	n := len(*x)
	item := (*x)[n-1]
	*x = (*x)[:n-1]
	return item
}

// v, l, s := model index in vocabulary, level, state
func (self *ConnectedHMM) getScore(scores [][][]float64, l LevelIndex, v VocIndex, s StateIndex) (float64, bool) {
	modelStates := scores[l]
	if modelStates == nil {
		return 0, false
	}
	states := modelStates[v]
	if states == nil || len(states) <= int(s) {
		return 0, false
	}
	return states[s], true
}

// inner/frequently used function, debug if panic
func (self *ConnectedHMM) setScore(scores [][][]float64, l LevelIndex, v VocIndex, s StateIndex, score float64) {
	modelStates := scores[l]
	if modelStates == nil {
		modelStates = make([][]float64, len(self.HMMs), len(self.HMMs))
		scores[l] = modelStates
		if s == 0 {
			modelStates[v] = []float64{score}
			return
		} else {
			panic(fmt.Sprintf("ConnectedHMM: state should start from zero for new level, v/l/s = %d/%d/%d", v, l, s))
		}
	}
	states := modelStates[v]
	if states == nil {
		if s == 0 {
			modelStates[v] = []float64{score}
			return
		} else {
			panic(fmt.Sprintf("ConnectedHMM: state should start from zero for new level, v/l/s = %d/%d/%d", v, l, s))
		}
	}
	if len(states) < int(s) {
		panic("ConnectedHMM: state should increase one by one")
	}
	if len(states) == int(s) {
		modelStates[v] = append(states, score)
		return
	}
	states[s] = score
	return
}

func (self *ConnectedHMM) newStateTrackingMatrix(L LevelIndex, V VocIndex) [][][]int {
	var lastStartTs [][][]int = make([][][]int, L, L)
	for l := LevelIndex(0); l < L; l++ {
		lastStartTs[l] = make([][]int, V, V)
		for v := VocIndex(0); v < V; v++ {
			num := self.HMMs[v].NumState
			lastStartTs[l][v] = make([]int, num, num)
		}
	}
	return lastStartTs
}

func (self *ConnectedHMM) newModelTrackingMatrix(T int, L LevelIndex, V VocIndex) [][][]*ModelTrack {
	var timeLevelVocPrevious [][][]*ModelTrack = make([][][]*ModelTrack, T, T)
	for t := 0; t < T; t++ {
		timeLevelVocPrevious[t] = make([][]*ModelTrack, L, L)
		for l := LevelIndex(0); l < L; l++ {
			timeLevelVocPrevious[t][l] = make([]*ModelTrack, V, V)
		}
	}
	return timeLevelVocPrevious
}

func (self *ConnectedHMM) getBestScore(l LevelIndex, v VocIndex, ds StateIndex, lastScores [][][]float64, hmm *HMM) (bool, float64, StateIndex) {
	var score float64 = 0
	var bestSrc StateIndex = 0
	var valid bool = false
	for _, ss := range hmm.GetSourceStates(int(ds)) {
		tmp, ok := self.getScore(lastScores, l, v, StateIndex(ss))
		if !ok {
			continue
		}
		tlpr, ok := hmm.GetLogTransitionProbability(int(ss), int(ds))
		if !ok {
			continue
		}
		tmp += tlpr
		if tmp > score || !valid {
			score = tmp
			bestSrc = StateIndex(ss)
			valid = true
		}
	}
	return valid, score, bestSrc
}

func (self *ConnectedHMM) ViterbiDecode(seq []core.DenseVector, maxModels int, nBest int) []*DecodedSequence {
	var T int = len(seq)
	var V VocIndex = VocIndex(len(self.HMMs))
	var timeLevelVocPrevious [][][]*ModelTrack = self.newModelTrackingMatrix(T, LevelIndex(maxModels), V) // at t, a new model v of level l starts, its ancient is tracked at (t, l, v)
	var lastScores [][][]float64 = make([][][]float64, maxModels, maxModels)                              // at t-1, score of state s of model v of level l is (l, v, s)
	var lastStartTs [][][]int = self.newStateTrackingMatrix(LevelIndex(maxModels), V)                     // at t-1, start time of state s of model v of level l is (l, v, s)
	// init
	// fmt.Println(0, seq[0])
	for v := VocIndex(0); v < V; v++ {
		hmm := self.HMMs[v]
		self.setScore(lastScores, LevelIndex(0), v, StateIndex(0), hmm.GetLogInitStateProbability(seq[0]))
	}
	// decoding
	for t := 1; t < T; t++ {
		// fmt.Println(t, seq[t])
		var curScores [][][]float64 = make([][][]float64, maxModels, maxModels)
		var curStartTs [][][]int = self.newStateTrackingMatrix(LevelIndex(maxModels), V)

		o := seq[t]
		lpr_s_v := make([]core.DenseVector64, V, V)
		for v := VocIndex(0); v < V; v++ {
			lpr_s_v[v] = self.HMMs[v].GetLogStateProbabilities(o)
		}
		var levelValid bool = false
		for l := LevelIndex(0); l < LevelIndex(maxModels); l++ {
			for v := VocIndex(0); v < V; v++ {
				hmm := self.HMMs[v]
				lpr_s := lpr_s_v[v]
				// boundary
				var initValid bool = false
				for ds := StateIndex(0); ds < 1; ds++ {
					var score float64
					var bestSrc StateIndex
					initValid, score, bestSrc = self.getBestScore(l, v, ds, lastScores, hmm)
					var lastV VocIndex = -1
					if l > 0 {
						for sv := VocIndex(0); sv < V; sv++ {
							sHMM := self.HMMs[sv]
							var valid bool
							var tmp float64
							var src StateIndex
							valid, tmp, src = self.getBestScore(l-1, sv, StateIndex(sHMM.NumState), lastScores, sHMM)
							if !valid {
								continue
							}
							if !initValid || tmp > score {
								score = tmp
								bestSrc = src
								lastV = sv
								initValid = true
							}
						}
					}
					if lastV != -1 {
						// model changed
						timeLevelVocPrevious[t][l][v] = &ModelTrack{
							LastV:      lastV,
							LastStartT: lastStartTs[l-1][lastV][bestSrc],
						}
						curStartTs[l][v][ds] = t
					} else {
						// model unchanged
						curStartTs[l][v][ds] = lastStartTs[l][v][bestSrc]
					}
					if initValid {
						self.setScore(curScores, l, v, ds, score+lpr_s[ds])
					}
				}
				if !initValid {
					break // if ds==0 is not achievable, then ds+ are NOT
				}
				levelValid = true
				// non-boundary: assuming the states spend one by one
				for ds := StateIndex(1); ds < StateIndex(hmm.NumState); ds++ {
					var stateValid = false
					var score float64 = 0
					var bestSrc StateIndex = 0
					stateValid, score, bestSrc = self.getBestScore(l, v, ds, lastScores, hmm)
					curStartTs[l][v][ds] = lastStartTs[l][v][bestSrc]
					if stateValid {
						self.setScore(curScores, l, v, ds, score+lpr_s[ds])
					} else {
						break // if ds is not achievalbe, then ds+ are NOT
					}
				}
			} // end of looping vocabulary
			if !levelValid {
				break // if a model is not achievable, other models are NOT
			}
		} // end of looping levels
		if !levelValid {
			break // if a level is not achievable, higher levels are NOT
		}
		lastScores = curScores
		lastStartTs = curStartTs
	} // end of looping time
	// ending
	pQueue := &ScoreEndStatePQueue{}
	heap.Init(pQueue)
	for v := VocIndex(0); v < V; v++ {
		hmm := self.HMMs[v]
		for l := LevelIndex(0); l < LevelIndex(maxModels); l++ {
			var score float64 = 0
			var valid bool = false
			var bestSrc StateIndex = 0
			valid, score, bestSrc = self.getBestScore(l, v, StateIndex(hmm.NumState), lastScores, hmm)
			if valid {
				item := &ScoreEndStateItem{Score: score, Level: l, Model: v, State: bestSrc}
				heap.Push(pQueue, item)
			}
		}
	}
	// get result
	var result []*DecodedSequence = make([]*DecodedSequence, 0)
	for n := 0; pQueue.Len() > 0 && n < nBest; n++ {
		item := heap.Pop(pQueue).(*ScoreEndStateItem)
		seqs := make([]VocIndex, item.Level+1)
		tracks := make([]*ModelTrack, item.Level+1)
		startT := lastStartTs[item.Level][item.Model][item.State]
		model := item.Model
		seqs[item.Level] = model
		tracks[item.Level] = &ModelTrack{LastV: model, LastStartT: startT}
		for l := item.Level - 1; l >= 0; l-- {
			track := timeLevelVocPrevious[startT][l+1][model]
			if track == nil {
				panic("ConnectedHMM: the tracking item is nil")
			}
			tracks[l] = track
			model = track.LastV
			seqs[l] = model
			startT = track.LastStartT
		}
		result = append(result, &DecodedSequence{Score: item.Score, Seqs: seqs, Tracks: tracks})
	}
	return result
}
