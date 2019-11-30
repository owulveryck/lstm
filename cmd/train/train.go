package main

import (
	"io"

	"github.com/owulveryck/lstm"
	"gorgonia.org/tensor"
)

func train(nn *lstm.LSTM, input io.ReadSeeker, config configuration) error {

	ds := newDataset(input, nn.Dict)

	x := tensor.NewDense(tensor.Float64, []int{len(nn.Dict), config.BatchSize})
	for epoch := 0; epoch < config.Epoch; epoch++ {
		if _, err := input.Seek(0, io.SeekStart); err != nil {
			return err
		}
		for {
			n, err := ds.read(x)
			if err != nil {
				if err == io.EOF {
					continue
				}
				return err
			}
			err = move(input, config.Step, int64(n))
			if err != nil {
				if err == io.EOF {
					continue
				}
				return err
			}
		}
	}
	return nil
}
