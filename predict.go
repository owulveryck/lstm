package lstm

import (
	"context"
	"io"

	"github.com/owulveryck/lstm/datasetter"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// basicReadWriter is a dummy structure that fufil the datasetter.ReadWriter interface
// Is it used to build a one step execution graph
type basicReadWriter struct {
	input  *G.Node
	step   int
	output *G.Node
}

func (b *basicReadWriter) ReadInputVector(g *G.ExprGraph) (*G.Node, error) {
	if b.step >= 1 {
		return nil, io.EOF
	}
	b.step++
	return b.input, nil
}

func (b *basicReadWriter) WriteComputedVector(n *G.Node) error {
	b.output = n
	return nil
}

// GetComputedVectors ...
func (b *basicReadWriter) GetComputedVectors() G.Nodes {
	return G.Nodes{b.output}
}

// Predict ...
func (m *Model) Predict(ctx context.Context, dataSet datasetter.Float32ReadWriter) error {
	hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(m.hiddenSize))
	cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(m.hiddenSize))
	lstm := m.newLSTM(hiddenT, cellT)
	// Create the inputVector
	inputBacking := make([]float32, m.inputSize)
	inputT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(m.inputSize), tensor.WithBacking(inputBacking))
	input := G.NewVector(lstm.g, tensor.Float32, G.WithName("input"), G.WithShape(m.inputSize), G.WithValue(inputT))
	// Create a dummy ReadWriter to build a basic computing graph
	dummySet := &basicReadWriter{
		input: input,
	}
	// We need an empty memory to start...
	prevHidden := G.NewVector(lstm.g, tensor.Float32, G.WithName("hₜ₋₁"), G.WithShape(m.hiddenSize), G.WithValue(hiddenT))
	prevCell := G.NewVector(lstm.g, tensor.Float32, G.WithName("Cₜ₋₁"), G.WithShape(m.hiddenSize), G.WithValue(cellT))
	// First pass to get update the hidden state and the cell according to the input
	hidden, cell, err := lstm.forwardStep(dummySet, prevHidden, prevCell, 0)
	if err != nil {
		return err
	}
	//g := lstm.g.SubgraphRoots(dataSet.GetComputedVectors()...)
	//machine := G.NewTapeMachine(g, G.ExecuteFwdOnly())
	machine := G.NewTapeMachine(lstm.g)
	for {
		inputValue, err := dataSet.Read()
		copy(input.Value().Data().([]float32), inputValue)
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		err = machine.RunAll()
		if err != nil {
			return err
		}
		machine.Reset()
		dataSet.Write(dummySet.output.Value().Data().([]float32))
		copy(prevHidden.Value().Data().([]float32), hidden.Value().Data().([]float32))
		copy(prevCell.Value().Data().([]float32), cell.Value().Data().([]float32))
	}
	return nil
}
