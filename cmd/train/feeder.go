package main

import (
	"context"
	"io"

	"gorgonia.org/tensor"
)

// feeder reads the input and feed the output channel with tensor according to the batch size passed in configuration
func feeder(ctx context.Context, dict []rune, input io.ReadSeeker, config configuration) (<-chan *tensor.Dense, <-chan error) {
	outputC := make(chan *tensor.Dense)
	errC := make(chan error, 1)
	go func() {
		defer close(errC)
		defer close(outputC)

		ds := newDataset(input, dict)
		x := tensor.NewDense(tensor.Float64, []int{len(dict), config.BatchSize})

		for {
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
			err = move(input, config.Step, int64(n))
			if err != nil {
				errC <- err
				return
			}
		}
	}()
	return outputC, errC
}
