package main

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestDataset_New(t *testing.T) {
	dict := []rune{'a', 'b', 'c', 'd'}
	testinput := `abcdabcdabcdabcdabcd`
	ds := newDataset(bytes.NewReader([]byte(testinput)), dict)
	if ds.reverse['a'] != 0 || ds.reverse['b'] != 1 || ds.reverse['c'] != 2 || ds.reverse['d'] != 3 {
		t.Fail()
	}
}

func TestDataset_Read(t *testing.T) {
	batchSize := 4
	dict := []rune{'a', 'b', 'c', 'd'}
	testinput := `abcdabcdabcdabcdabcd`
	x := tensor.NewDense(tensor.Float64, []int{len(dict), batchSize})
	ds := newDataset(bytes.NewReader([]byte(testinput)), dict)
	err := ds.read(x)
	if err != nil {
		t.Fatal(err)
	}
	val, ok := x.Data().([]float64)
	if !ok {
		t.Fail()
	}
	assert.Equal(t, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, val, "bad value")
}
