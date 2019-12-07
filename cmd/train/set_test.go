package main

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/owulveryck/lstm"
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
	if !reflect.DeepEqual(y[0].Value().Data().([]float64), []float64{1, 5, 9}) {
		t.Fatalf("y[0] is %v", y[0].Value().Data())
	}
	if !reflect.DeepEqual(y[1].Value().Data().([]float64), []float64{2, 6, 10}) {
		t.Fatalf("y[1] is %v", y[1].Value().Data())
	}
	if !reflect.DeepEqual(y[2].Value().Data().([]float64), []float64{3, 7, 11}) {
		t.Fatalf("y[2] is %v", y[2].Value().Data())
	}
}

func TestSetLSTMValues(t *testing.T) {
	vectorSize := 3
	hiddenSize := 2
	batchSize := 3
	nn := lstm.NewLSTM(vectorSize, hiddenSize)
	network := lstm.NewNetwork(nn, 4)
	y := make([]*gorgonia.Node, batchSize)
	for i := 0; i < batchSize; i++ {
		y[i] = gorgonia.NewVector(nn.G, gorgonia.Float64, gorgonia.WithName(fmt.Sprintf("yy%v", i)),
			gorgonia.WithShape(vectorSize))
	}

	xT := tensor.NewDense(
		tensor.Float64,
		[]int{vectorSize, batchSize + 1},
		tensor.WithBacking(
			[]float64{
				0, 1, 2, 3,
				4, 5, 6, 7,
				8, 9, 10, 11}))

	err := setInputValues(network, y, xT)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(y[0].Value().Data().([]float64), []float64{1, 5, 9}) {
		t.Fatalf("y[0] is %v", y[0].Value().Data())
	}
	if !reflect.DeepEqual(y[1].Value().Data().([]float64), []float64{2, 6, 10}) {
		t.Fatalf("y[1] is %v", y[1].Value().Data())
	}
	if !reflect.DeepEqual(y[2].Value().Data().([]float64), []float64{3, 7, 11}) {
		t.Fatalf("y[2] is %v", y[2].Value().Data())
	}
	if !reflect.DeepEqual(network.X[0].Value().Data().([]float64), []float64{0, 4, 8}) {
		t.Fatalf("X[0] is %v", network.X[0].Value().Data())
	}
	if !reflect.DeepEqual(network.X[1].Value().Data().([]float64), []float64{1, 5, 9}) {
		t.Fatalf("X[1] is %v", network.X[1].Value().Data())
	}
	if !reflect.DeepEqual(network.X[2].Value().Data().([]float64), []float64{2, 6, 10}) {
		t.Fatalf("X[2] is %v", network.X[2].Value().Data())
	}
	if !reflect.DeepEqual(network.H[0].Value().Data().([]float64), make([]float64, network.H[0].Shape()[0])) {
		t.Fatalf("H[0] is %v", network.H[0].Value().Data())
	}
	if !reflect.DeepEqual(network.C[0].Value().Data().([]float64), make([]float64, network.C[0].Shape()[0])) {
		t.Fatalf("C[0] is %v", network.C[0].Value().Data())
	}
}
