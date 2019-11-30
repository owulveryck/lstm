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
// it returns the number of bytes read
func (d *dataset) read(x *tensor.Dense) (int, error) {
	rdr := bufio.NewReader(d.input)
	runes := make([]rune, x.Shape()[0])
	bytesRead := 0
	var err error
	var n int
	for i := 0; i < x.Shape()[0]; i++ {
		runes[i], n, err = rdr.ReadRune()
		if err != nil {
			return bytesRead, err
		}
		bytesRead += n
	}
	// free the tensor
	for i := 0; i < len(x.Data().([]float64)); i++ {
		x.Data().([]float64)[i] = 0
	}
	for i, run := range runes {
		x.SetAt(float64(1), i, d.reverse[run])
	}
	return bytesRead, nil
}
