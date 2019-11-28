package main

import (
	"io"

	"github.com/owulveryck/lstm"
)

func train(nn *lstm.LSTM, input io.ReadSeeker, config configuration) error {

	for epoch := 0; epoch < config.Epoch; epoch++ {
		if _, err := input.Seek(0, io.SeekStart); err != nil {
			return err
		}
	}
	return nil
}
