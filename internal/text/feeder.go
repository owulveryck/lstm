package text

import (
	"bytes"
	"context"
	"sync"

	"gorgonia.org/tensor"
)

// Feeder reads the input and feed the output channel with tensor according to the batch size passed in configuration.
// The output tensor shape are len(dict)xbatchSize.
// Each column of this matrix is a rune, one-hot-encoded according to the index of the dict array.
func Feeder(ctx context.Context, dict []rune, input *bytes.Reader, batchSize, step int) (<-chan *tensor.Dense, <-chan error) {
	outputC := make(chan *tensor.Dense)
	errC := make(chan error, 1)

	var tensorPool = sync.Pool{
		New: func() interface{} {
			return tensor.NewDense(tensor.Float64, []int{len(dict), batchSize})
		},
	}
	go func() {
		defer close(errC)
		defer close(outputC)

		ds := newDataset(input, dict)
		//x := tensor.NewDense(tensor.Float64, []int{len(dict), batchSize})

		for {
			x := tensorPool.Get().(*tensor.Dense)
			n, err := ds.read(x)
			if err != nil {
				errC <- err
				return
			}
			select {
			case outputC <- x:
			case <-ctx.Done():
				return
			}
			tensorPool.Put(x)
			err = move(input, step, int64(n))
			if err != nil {
				errC <- err
				return
			}
		}
	}()
	return outputC, errC
}
