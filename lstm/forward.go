package lstm

import (
	"fmt"
	"io/ioutil"

	"github.com/chewxy/gorgonia/tensor"
	G "gorgonia.org/gorgonia"
)

type lstmOut struct {
	hiddens G.Nodes
	cells   G.Nodes

	probs *G.Node
}

func (m *Model) forwardPass(srcIndex int, prev *lstmOut) (retVal *lstmOut, err error) {
	var prevHiddens G.Nodes
	var prevCells G.Nodes

	if prev == nil {
		prevHiddens = m.prevHiddens
		prevCells = m.prevCells
	} else {
		prevHiddens = prev.hiddens
		prevCells = prev.cells
	}

	inputVector := m.inputVector
	inputValue := make([]float32, m.inputSize)
	inputValue[srcIndex] = 1.0
	inputVectorT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(m.inputSize), tensor.WithBacking(inputValue))

	var hiddens, cells G.Nodes
	for i, l := range m.ls {
		if i == 0 {
			inputVector = G.NewVector(m.g, tensor.Float32, G.WithName("InputVector0"), G.WithShape(m.inputSize), G.WithValue(inputVectorT))
		} else {
			inputVector = hiddens[i-1]
		}

		prevHidden := prevHiddens[i]
		prevCell := prevCells[i]

		var h0, h1, inputGate *G.Node
		h0 = G.Must(G.Mul(l.wix, inputVector))
		h1 = G.Must(G.Mul(l.wih, prevHidden))
		inputGate = G.Must(G.Sigmoid(G.Must(G.Add(G.Must(G.Add(h0, h1)), l.biasI))))

		var h2, h3, forgetGate *G.Node
		h2 = G.Must(G.Mul(l.wfx, inputVector))
		h3 = G.Must(G.Mul(l.wfh, prevHidden))
		forgetGate = G.Must(G.Sigmoid(G.Must(G.Add(G.Must(G.Add(h2, h3)), l.biasF))))

		var h4, h5, outputGate *G.Node
		h4 = G.Must(G.Mul(l.wox, inputVector))
		h5 = G.Must(G.Mul(l.woh, prevHidden))
		outputGate = G.Must(G.Sigmoid(G.Must(G.Add(G.Must(G.Add(h4, h5)), l.biasO))))

		var h6, h7, cellWrite *G.Node
		h6 = G.Must(G.Mul(l.wcx, inputVector))
		h7 = G.Must(G.Mul(l.wch, prevHidden))
		cellWrite = G.Must(G.Tanh(G.Must(G.Add(G.Must(G.Add(h6, h7)), l.biasC))))

		// cell activations
		var retain, write, cell, hidden *G.Node
		retain = G.Must(G.HadamardProd(forgetGate, prevCell))
		write = G.Must(G.HadamardProd(inputGate, cellWrite))
		cell = G.Must(G.Add(retain, write))
		hidden = G.Must(G.HadamardProd(outputGate, G.Must(G.Tanh(cell))))

		hiddens = append(hiddens, hidden)
		cells = append(cells, cell)
	}

	lastHidden := hiddens[len(hiddens)-1]
	var output *G.Node
	if output, err = G.Mul(m.whd, lastHidden); err == nil {
		if output, err = G.Add(output, m.biasD); err != nil {
			G.WithName("LAST HIDDEN")(lastHidden)
			ioutil.WriteFile("err.dot", []byte(lastHidden.RestrictedToDot(3, 10)), 0644)
			panic(fmt.Sprintf("ERROR: %v", err))
		}
	}

	var probs *G.Node
	probs = G.Must(G.SoftMax(output))

	retVal = &lstmOut{
		hiddens: hiddens,
		cells:   cells,
		probs:   probs,
	}
	return
}
