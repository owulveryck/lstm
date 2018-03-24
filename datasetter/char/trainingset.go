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
	rs        io.Reader
	runeToIdx func(r rune) (int, error)
	vocabSize int
	step      int
}

// Section ...
type Section struct {
	sentence  []int
	output    G.Nodes
	vocabSize int
	offset    int
}

// NewTrainingSet from a ReadSeeker
func NewTrainingSet(rs io.Reader, runeToIdx func(r rune) (int, error), vocabSize, step int) *TrainingSet {
	return &TrainingSet{
		rs:        rs,
		vocabSize: vocabSize,
		step:      step,
		runeToIdx: runeToIdx,
	}
}

// ReadInputVector returns the input vector until it reach the penultimate rune
// the ultimate rune is not used as input within the current section as an input
func (s *Section) ReadInputVector(g *G.ExprGraph) (*G.Node, error) {
	if s.offset == len(s.sentence) {
		return nil, io.EOF
	}
	backend := make([]float32, s.vocabSize)
	backend[s.sentence[s.offset]] = 1
	inputTensor := tensor.New(tensor.WithShape(s.vocabSize), tensor.WithBacking(backend))
	node := G.NewVector(g, tensor.Float32, G.WithName(fmt.Sprintf("input_%v", s.offset)), G.WithShape(s.vocabSize), G.WithValue(inputTensor))
	s.offset++
	return node, nil
}

// WriteComputedVector add the computed vectors to the output
func (s *Section) WriteComputedVector(n *G.Node) error {
	s.output = append(s.output, n)
	return nil
}

// GetComputedVectors ..
func (s *Section) GetComputedVectors() G.Nodes {
	return s.output
}

// GetExpectedValue returns the encoded value of the rune next to the one present at offset
func (s *Section) GetExpectedValue(offset int) (int, error) {
	return s.sentence[offset], nil
}

// GetTrainingSet is returning the self object with the
// currentSection field adapted to it contains a full set of runes
func (t *TrainingSet) GetTrainingSet() (*Section, error) {
	section := &Section{
		vocabSize: t.vocabSize,
		offset:    0,
		sentence:  make([]int, t.step),
	}
	buf := bufio.NewReader(t.rs)
	for i := 0; i < t.step; i++ {
		rne, _, err := buf.ReadRune()
		idx, err := t.runeToIdx(rne)
		if err != nil {
			return nil, err
		}
		section.sentence[i] = idx
	}
	// TODO: set the offset to currentoffset+step
	return section, nil
}
