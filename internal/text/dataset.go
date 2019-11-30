package text

import (
	"bytes"

	"gorgonia.org/tensor"
)

type dataset struct {
	input   *bytes.Reader
	dict    []rune
	reverse map[rune]int
}

func newDataset(input *bytes.Reader, dict []rune) *dataset {
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
	rdr := d.input
	runes := make([]rune, x.Shape()[1])
	bytesRead := 0
	var err error
	var n int
	for i := 0; i < x.Shape()[1]; i++ {
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
		x.SetAt(float64(1), d.reverse[run], i)
	}
	return bytesRead, nil
}
