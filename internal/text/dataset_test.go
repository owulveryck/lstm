package text

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestDataset_New(t *testing.T) {
	dict := []rune{'a', 'b', 'c', 0x2318}
	testinput := `abc⌘abc⌘abc⌘abc⌘abc⌘`
	ds := newDataset(bytes.NewReader([]byte(testinput)), dict)
	if ds.reverse['a'] != 0 || ds.reverse['b'] != 1 || ds.reverse['c'] != 2 || ds.reverse['⌘'] != 3 {
		t.Fail()
	}
}

func TestDataset_Read_batchsize4(t *testing.T) {
	batchSize := 4
	dict := []rune{'a', 'b', 'c', 0x2318}
	testinput := `abc⌘abc⌘abc⌘abc⌘abc⌘`
	x := tensor.NewDense(tensor.Float64, []int{len(dict), batchSize})
	ds := newDataset(bytes.NewReader([]byte(testinput)), dict)
	n, err := ds.read(x)
	if err != nil {
		t.Fatal(err)
	}
	if n != 6 {
		t.Fail()
	}
	val, ok := x.Data().([]float64)
	if !ok {
		t.Fail()
	}
	assert.Equal(t, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, val, "bad value")
}
func TestDataset_Read_batchsize5(t *testing.T) {
	batchSize := 5
	dict := []rune{'a', 'b', 'c', 0x2318}
	testinput := `abc⌘abc⌘abc⌘abc⌘abc⌘`
	x := tensor.NewDense(tensor.Float64, []int{len(dict), batchSize})
	ds := newDataset(bytes.NewReader([]byte(testinput)), dict)
	n, err := ds.read(x)
	if err != nil {
		t.Fatal(err)
	}
	if n != 6 {
		t.Fail()
	}
	val, ok := x.Data().([]float64)
	if !ok {
		t.Fail()
	}
	assert.Equal(t, []float64{1, 0, 0, 0, 1,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0}, val, "bad value")
}
