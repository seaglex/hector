package core

import (
	"errors"
	"fmt"
	"strings"
)

// for input/output variables, simple calculation
type DenseVector []float32

// For intermediate variables, complex calculation
type DenseVector64 []float64

// Instance Method
func (self DenseVector) Dimension() int {
	return len(self)
}

func (self DenseVector64) Dimension() int {
	return len(self)
}

func (self DenseVector64) MultiplyElemWise(vec DenseVector64) {
	for n, v := range vec {
		(self)[n] *= v
	}
}

func (self DenseVector64) Scale(scale float64) {
	for n := 0; n < self.Dimension(); n++ {
		(self)[n] *= scale
	}
}

// static methods
func Sum(vec DenseVector64) float64 {
	var total float64 = 0
	for _, x := range vec {
		total += x
	}
	return total
}

func MultiplyElemWise(x, y DenseVector64) DenseVector64 {
	if len(x) != len(y) {
		panic(errors.New("DenseVec: dimension mismatches"))
	}
	z := make(DenseVector64, len(x), len(x))
	for n, v := range x {
		z[n] = v * y[n]
	}
	return z
}

func GetSparseMatColumn(m *Matrix, dim int, column int) DenseVector64 {
	x := make(DenseVector64, dim, dim)
	for n, vec := range m.Data {
		v := vec.GetValue(int64(column))
		x[n] = v
	}
	return x
}

func MultiplyVecSparseMat(x DenseVector64, m *Matrix) DenseVector64 {
	y := make(DenseVector64, x.Dimension(), x.Dimension())
	for r, vec := range m.Data {
		for c, v := range vec.Data {
			y[c] += v * x[r]
		}
	}
	return y
}

func MultiplySparseMatVec(m *Matrix, x DenseVector64) DenseVector64 {
	y := make(DenseVector64, x.Dimension(), x.Dimension())
	for n, vec := range m.Data {
		var s float64 = 0
		for i, v := range vec.Data {
			s += v * x[i]
		}
		y[n] = s
	}
	return y
}

func Mat2String(x *Matrix) string {
	lines := []string{}
	for c, vec := range x.Data {
		lines = append(lines, fmt.Sprint(c, vec.Data))
	}
	return strings.Join(lines, ";")
}

func CloneMatrix(x *Matrix) *Matrix {
	y := NewMatrix()
	for r, vx := range x.Data {
		vy := NewVector()
		for c, val := range vx.Data {
			vy.Data[c] = val
		}
		y.Data[r] = vy
	}
	return y
}
