package char

import (
	"bytes"
	"fmt"
	"io"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Prediction is the based type that can be used as a training dataset
type Prediction struct {
	input      *bytes.Buffer
	runeToIdx  func(r rune) (int, error)
	sampleSize int
	generated  int
	vocabSize  int
	output     G.Nodes
}

// NewPrediction return an object suitable for the LSTM
func NewPrediction(input string, runeToIdx func(r rune) (int, error), sampleSize, vocabSize int) *Prediction {
	return &Prediction{
		input:      bytes.NewBufferString(input),
		runeToIdx:  runeToIdx,
		sampleSize: sampleSize,
		vocabSize:  vocabSize,
	}
}

// ReadInputVector ...
func (p *Prediction) ReadInputVector(g *G.ExprGraph) (*G.Node, error) {
	rn, _, err := p.input.ReadRune()
	if err != nil && err != io.EOF {
		return nil, err
	}
	if err == io.EOF && p.generated < p.sampleSize {
		p.generated++
		return p.output[len(p.output)-1], nil
	}
	if p.generated >= p.sampleSize {
		return nil, io.EOF
	}
	backend := make([]float32, p.vocabSize)
	idx, err := p.runeToIdx(rn)
	if err != nil {
		return nil, err
	}
	backend[idx] = 1
	inputTensor := tensor.New(tensor.WithShape(p.vocabSize), tensor.WithBacking(backend))
	node := G.NewVector(g, tensor.Float32, G.WithName(fmt.Sprintf("input_%v", p.generated)), G.WithShape(p.vocabSize), G.WithValue(inputTensor))
	return node, nil
}

// WriteComputedVector ...
func (p *Prediction) WriteComputedVector(n *G.Node) error {
	// TODO: apply a function for the classification in order to get a vector with a uniq entry to one
	p.output = append(p.output, n)
	return nil
}

// GetComputedVectors ...
func (p *Prediction) GetComputedVectors() G.Nodes {
	return p.output
}
