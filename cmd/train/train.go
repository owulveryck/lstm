package main

import (
	"bytes"
	"context"
	"io"
	"io/ioutil"

	"github.com/owulveryck/lstm"
	"github.com/owulveryck/lstm/internal/text"
	"gorgonia.org/tensor"
)

func run(nn *lstm.LSTM, input io.Reader, config configuration) error {
	// Create a new context
	ctx := context.Background()
	// Create a new context, with its cancellation function
	// from the original context
	ctx, cancel := context.WithCancel(ctx)

	content, err := ioutil.ReadAll(input)
	if err != nil {
		return err
	}
	rdr := bytes.NewReader(content)
	for i := 0; i < config.Epoch; i++ {
		_, err := rdr.Seek(0, io.SeekStart)
		if err != nil {
			return err
		}
		feedC, errC := text.Feeder(ctx, nn.Dict, rdr, config.BatchSize, config.Step)

		for x := range feedC {
			err := train(nn, x)
			if err != nil {
				cancel()
				return err
			}
		}
		if err := <-errC; err != nil {
			if err != io.EOF {
				return err
			}
		}
	}
	return nil
}

func train(nn *lstm.LSTM, x *tensor.Dense) error {
	return nil
}
