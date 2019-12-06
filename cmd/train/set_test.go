package main

import (
	"fmt"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestSetYValues(t *testing.T) {
	vectorSize := 3
	batchSize := 3
	g := gorgonia.NewGraph()
	err := setValues(nil, nil)
	if err == nil {
		t.Fail()
	}
	xT := tensor.NewDense(tensor.Float64, []int{vectorSize})
	err = setValues(nil, xT)
	if err == nil {
		t.Fail()
	}
	xT = tensor.NewDense(tensor.Float64, []int{2, 2})
	err = setValues(nil, xT)
	if err == nil {
		t.Fail()
	}
	xT = tensor.NewDense(
		tensor.Float64,
		[]int{vectorSize, batchSize + 1},
		tensor.WithBacking(
			[]float64{
				0, 1, 2, 3,
				4, 5, 6, 7,
				8, 9, 10, 11}))

	y := make([]*gorgonia.Node, 2)
	err = setValues(y, xT)
	if err == nil {
		t.Fail()
	}
	y = make([]*gorgonia.Node, batchSize)
	for i := 0; i < batchSize; i++ {
		y[i] = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithName(fmt.Sprintf("yy%v", i)),
			gorgonia.WithShape(vectorSize, vectorSize))
	}
	err = setValues(y, xT)
	if err == nil {
		t.Fail()
	}
	for i := 0; i < batchSize; i++ {
		y[i] = gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithName(fmt.Sprintf("yy%v", i)),
			gorgonia.WithShape(vectorSize+1))
	}
	err = setValues(y, xT)
	if err == nil {
		t.Fail()
	}

	y = generateY(g, vectorSize, batchSize)
	vT, err := xT.Slice(nil, makeRS(1, batchSize+1))
	if err != nil {
		t.Fatal(err)
	}
	err = setValues(y, vT.Materialize().(*tensor.Dense))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < batchSize; i++ {
		t.Log(y[i].Value().Data().([]float64))
	}

}
