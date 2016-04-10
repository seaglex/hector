package eval

import (
	"math"
	"sort"
)

type LabelPrediction struct {
	Prediction float64
	Label      int
}

type RealPrediction struct { // Real valued
	Prediction float64
	Value      float64
}

type By func(p1, p2 *LabelPrediction) bool

type labelPredictionSorter struct {
	predictions []*LabelPrediction
	by          By
}

func (s *labelPredictionSorter) Len() int {
	return len(s.predictions)
}

func (s *labelPredictionSorter) Swap(i, j int) {
	s.predictions[i], s.predictions[j] = s.predictions[j], s.predictions[i]
}

func (s *labelPredictionSorter) Less(i, j int) bool {
	return s.by(s.predictions[i], s.predictions[j])
}

func (by By) Sort(predictions []*LabelPrediction) {
	sorter := &labelPredictionSorter{
		predictions: predictions,
		by:          by,
	}
	sort.Sort(sorter)
}

func AUC(predictions0 []*LabelPrediction) float64 {
	predictions := []*LabelPrediction{}
	for _, pred := range predictions0 {
		predictions = append(predictions, pred)
	}
	prediction := func(p1, p2 *LabelPrediction) bool {
		return p1.Prediction > p2.Prediction
	}

	By(prediction).Sort(predictions)

	var count float64 = float64(len(predictions))
	var lastPred float64 = 0
	var acc float64 = 0
	var acc_neg float64 = 0
	var acc_pos float64 = 0
	pn := 0.0
	nn := 0.0
	for i, lp := range predictions {
		if i > 0 && lastPred != lp.Prediction {
			acc += pn * (count - acc_neg - nn*0.5)
			acc_pos += pn
			acc_neg += nn
			pn = 0
			nn = 0
		}
		if lp.Label > 0 {
			pn += 1
		} else {
			nn += 1
		}
		lastPred = lp.Prediction
	}
	acc += pn * (count - acc_neg - nn*0.5)
	acc_pos += pn
	acc_neg += nn
	if acc_pos*acc_neg == 0.0 {
		return 0.5
	}
	return acc/(acc_pos*acc_neg) - acc_pos/acc_neg
}

func RMSE(predictions []*LabelPrediction) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		ret += (float64(pred.Label) - pred.Prediction) * (float64(pred.Label) - pred.Prediction)
		n += 1.0
	}

	return math.Sqrt(ret / n)
}

func ErrorRate(predictions []*LabelPrediction) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		if (float64(pred.Label)-0.5)*(pred.Prediction-0.5) < 0 {
			ret += 1.0
		}
		n += 1.0
	}
	return ret / n
}

func RegRMSE(predictions []*RealPrediction) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		ret += (pred.Value - pred.Prediction) * (pred.Value - pred.Prediction)
		n += 1.0
	}

	return math.Sqrt(ret / n)
}

func KSTest(predictions0 []*LabelPrediction) float64 {
	predictions := make([]*LabelPrediction, len(predictions0), len(predictions0))
	for n, pred := range predictions0 {
		predictions[n] = pred
	}
	prediction := func(p1, p2 *LabelPrediction) bool {
		return p1.Prediction > p2.Prediction
	}
	By(prediction).Sort(predictions)

	var numPos float64 = 0.0
	var numNeg float64 = 0.0
	for _, pred := range predictions {
		if pred.Label > 0 {
			numPos += 1.0
		} else {
			numNeg += 1.0
		}
	}

	if numPos == 0 || numNeg == 0 {
		return 0.0
	}
	var tp float64 = 0.0 // true positive
	var fp float64 = 0.0 // false positive
	var diff float64 = 0.0
	scale := numPos / numNeg
	for _, lp := range predictions {
		if lp.Label > 0 {
			tp += 1.0
		} else {
			fp += 1.0
		}
		diff = math.Max(diff, tp-fp*scale)
	}
	return diff / numPos
}
