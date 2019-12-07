package main

import (
	"errors"

	"github.com/owulveryck/lstm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func setValues(n []*gorgonia.Node, xT *tensor.Dense) error {
	if xT == nil {
		return errors.New("xT is nil")
	}
	if xT.Dims() != 2 {
		return errors.New("xT is not a matrix")
	}
	if xT.Shape()[1] != len(n) {
		return errors.New("Expected len(n) to be batchsize")
	}
	for i := 0; i < len(n); i++ {
		if n[i].Dims() != 1 {
			return errors.New("Expected n to be a vector")

		}
		if n[i].Shape()[0] != xT.Shape()[0] {
			return errors.New("Expected node and xT's row are unequal")
		}
		vT, err := xT.Slice(nil, makeRS(i, i+1))
		if err != nil {
			return err
		}
		gorgonia.Let(n[i], vT.Materialize())
	}
	return nil
}

func setLSTMValues(model *lstm.Network, y []*gorgonia.Node, xT *tensor.Dense) error {
	viewY, err := xT.Slice(nil, makeRS(1, len(y)+1))
	if err != nil {
		return err
	}
	err = setValues(y, viewY.Materialize().(*tensor.Dense))
	if err != nil {
		return err
	}
	viewX, err := xT.Slice(nil, makeRS(0, len(model.X)))
	if err != nil {
		return err
	}
	err = setValues(model.X, viewX.Materialize().(*tensor.Dense))
	if err != nil {
		return err
	}
	if model.H[0].Value() == nil {
		shape := model.H[0].Shape()
		hT := tensor.NewDense(tensor.Float64, shape, tensor.WithBacking(make([]float64, shape[0])))
		gorgonia.Let(model.H[0], hT)
	} else {
	}
	if model.C[0].Value() == nil {
		shape := model.C[0].Shape()
		hT := tensor.NewDense(tensor.Float64, shape, tensor.WithBacking(make([]float64, shape[0])))
		gorgonia.Let(model.C[0], hT)
	} else {
	}
	return nil
}

type rs struct {
	start, end, step int
}

func (s rs) Start() int { return s.start }
func (s rs) End() int   { return s.end }
func (s rs) Step() int  { return s.step }

// makeRS creates a ranged slice. It takes an optional step param.
func makeRS(start, end int) rs {
	step := 1
	return rs{
		start: start,
		end:   end,
		step:  step,
	}
}
