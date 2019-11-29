package main

import (
	"bufio"
	"io"

	"gorgonia.org/tensor"
)

type dataset struct {
	input   io.ReadSeeker
	dict    []rune
	reverse map[rune]int
}

func newDataset(input io.ReadSeeker, dict []rune) *dataset {
	reverse := make(map[rune]int, len(dict))
	for i := 0; i < len(dict); i++ {
		reverse[dict[i]] = i
	}
	return &dataset{
		input:   input,
		dict:    dict,
		reverse: reverse,
	}
}

// read values fron the input structure and make it fit into the x receiver
func (d *dataset) read(x *tensor.Dense) error {
	rdr := bufio.NewReader(d.input)
	runes := make([]rune, x.Shape()[0])
	var err error
	for i := 0; i < x.Shape()[1]; i++ {
		runes[i], _, err = rdr.ReadRune()
		if err != nil {
			return err
		}
	}
	// free the tensor
	for i := 0; i < len(x.Data().([]float64)); i++ {
		x.Data().([]float64)[i] = 0
	}
	for i, run := range runes {
		x.SetAt(float64(1), i, d.reverse[run])
	}
	return nil
}
