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
	y := generateY(nn.G, nn.VectorSize, config.BatchSize)
	solver := gorgonia.NewRMSPropSolver(
		gorgonia.WithLearnRate(config.Learnrate),
		gorgonia.WithL2Reg(config.L2reg),
		gorgonia.WithClip(config.ClipVal))
	cost, err := model.CrossEntropy(y)
	if err != nil {
		return err
	}
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)
	_, err = gorgonia.Grad(cost, nn.Learnables()...)
	if err != nil {
		return err
	}
	vm := gorgonia.NewTapeMachine(nn.G, gorgonia.BindDualValues(nn.Learnables()...))
	defer vm.Close()
	for i := 0; i < config.Epoch; i++ {
		//		fmt.Printf("Epoch %v: ", i)
		_, err := rdr.Seek(0, io.SeekStart)
		if err != nil {
			return err
		}
		feedC, errC := text.Feeder(ctx, nn.Dict, rdr, config.BatchSize+1, config.Step)

		for xT := range feedC {

			err := setLSTMValues(model, y, xT)
			if err != nil {
				cancel()
				return err
			}
			err = vm.RunAll()
			if err != nil {
				cancel()
				return err
			}

			err = solver.Step(gorgonia.NodesToValueGrads(nn.Learnables()))
			if err != nil {
				cancel()
				return err
			}
			vm.Reset()
			fmt.Println(costVal)
		}
		if err := <-errC; err != nil {
			if err != io.EOF {
				return err
			}
		}
	}
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
