package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/owulveryck/lstm"
	"github.com/owulveryck/lstm/internal/text"
	"gorgonia.org/gorgonia"
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
	model := lstm.NewNetwork(nn, config.BatchSize)
	for i := 0; i < config.Epoch; i++ {
		_, err := rdr.Seek(0, io.SeekStart)
		if err != nil {
			return err
		}
		feedC, errC := text.Feeder(ctx, nn.Dict, rdr, config.BatchSize, config.Step)

		for x := range feedC {
			err := train(model, x)
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

func train(model *lstm.Network, x *tensor.Dense) error {
	return nil
}

func generateY(g *gorgonia.ExprGraph, vectorSize, batchSize int) []*gorgonia.Node {
	y := make([]*gorgonia.Node, batchSize)
	for i := 0; i < batchSize; i++ {
		y[i] = gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithName(fmt.Sprintf("yy%v", i)),
			gorgonia.WithShape(vectorSize))
	}
	return y
}
