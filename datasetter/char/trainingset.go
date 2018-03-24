package char

import (
	"bufio"
	"fmt"
	"io"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TrainingSet ...
type TrainingSet struct {
	rs             io.ReaderAt
	currentSection *io.SectionReader
	runeToIdx      func(r rune) (int, error)
	vocabSize      int
	idxToRune      func(idx int) (rune, error)
	outputStream   io.ReadWriter
	offset         int
	output         G.Nodes
	epoch          int
	maxEpoch       int
}

// NewTrainingSet from a ReaderAt
func NewTrainingSet(rs io.ReaderAt) *TrainingSet {
	return &TrainingSet{
		rs: rs,
	}
}

// ReadInputVector returns the input vector until it reach the penultimate rune
// the ultimate rune is not used as input within the current section as an input
func (t *TrainingSet) ReadInputVector(g *G.ExprGraph) (*G.Node, error) {
	b := bufio.NewReader(t.currentSection)
	r, size, err := b.ReadRune()
	if err != nil {
		return nil, err
	}
	t.offset += size
	// Check if the last rune is returning an error
	_, _, err = b.ReadRune()
	if err == io.EOF {
		return nil, err
	}
	// put the rune back in the stream
	err = b.UnreadRune()
	if err != nil {
		return nil, err
	}
	idx, err := t.runeToIdx(r)
	if err != nil {
		return nil, err
	}
	backend := make([]float32, t.vocabSize)
	backend[idx] = 1
	inputTensor := tensor.New(tensor.WithShape(t.vocabSize), tensor.WithBacking(backend))
	node := G.NewVector(g, tensor.Float32, G.WithName(fmt.Sprintf("input_%v", t.offset)), G.WithShape(t.vocabSize), G.WithValue(inputTensor))
	return node, nil
}

// WriteComputedVector add the computed vectors to the output
func (t *TrainingSet) WriteComputedVector(n *G.Node) error {
	t.output = append(t.output, n)
	return nil
}

// GetComputedVectors ..
func (t *TrainingSet) GetComputedVectors() G.Nodes {
	return t.output
}

// GetExpectedValue returns the encoded value of the rune next to the one present at offset
func (t *TrainingSet) GetExpectedValue(offset int) (int, error) {
	return 0, nil
}

// GetTrainingSet is returning the self object with the
// currentSection field adapted to it contains a full set of runes
func (t *TrainingSet) GetTrainingSet() (*TrainingSet, error) {

	return t, nil
}
