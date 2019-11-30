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

type network struct {
	lstm *lstm.LSTM
	x    []*gorgonia.Node
	h    []*gorgonia.Node
	c    []*gorgonia.Node
	y    []*gorgonia.Node
	cost *gorgonia.Node
}

func newNetwork(nn *lstm.LSTM, batchSize int) *network {
	vectorSize := nn.VectorSize
	hiddenSize := nn.HiddenSize
	x := make([]*gorgonia.Node, batchSize-1)
	y := make([]*gorgonia.Node, batchSize-1)
	h := make([]*gorgonia.Node, batchSize)
	c := make([]*gorgonia.Node, batchSize)
	h[0] = gorgonia.NewVector(nn.G, gorgonia.Float64, gorgonia.WithName("hₜ"),
		gorgonia.WithShape(hiddenSize))
	c[0] = gorgonia.NewVector(nn.G, gorgonia.Float64, gorgonia.WithName("cₜ"),
		gorgonia.WithShape(hiddenSize))
	for i := 0; i < batchSize-1; i++ {
		x[i] = gorgonia.NewVector(nn.G, gorgonia.Float64, gorgonia.WithName(fmt.Sprintf("xₜ+%v", i)),
			gorgonia.WithShape(vectorSize))
		h[i+1], c[i+1] = nn.NewCell(x[i], h[i], c[i])
		y[i] = nn.LogProb(h[i+1])
	}
	return &network{
		lstm: nn,
		x:    x,
		h:    h,
		c:    c,
		y:    y,
	}
}
