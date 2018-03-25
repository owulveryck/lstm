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
	rs        io.ReadSeeker
	runeToIdx func(r rune) (int, error)
	batchSize int
	vocabSize int
	step      int
	pass      int
}

// Section ...
type Section struct {
	sentence  []int
	output    G.Nodes
	vocabSize int
	offset    int
}

// NewTrainingSet from a ReadSeeker
func NewTrainingSet(rs io.ReadSeeker, runeToIdx func(r rune) (int, error), vocabSize, batchsize, step int) *TrainingSet {
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

// GetTrainingSet returns a pointer so a Section. It reads batchSize runes
// and add it to the returned section.
// The offset of the underlying io.ReadSeeker is set to the position it had
// when entering the function + step * runes * rune_size
// Any error in reading is returned
func (t *TrainingSet) GetTrainingSet() (*Section, error) {
	buf := bufio.NewReader(t.rs)
	// if we are not at the begining of the file,
	// we have done already a pass, then move the cursor
	// This is done at the begining of the file so whatever error could be return
	// If it was done at the end of the pass, any error would lead to
	// be interpreted by the caller of the func as "unable to provide a Section"
	// and the corresponding section would be discarded
	if t.pass != 0 {
		// move x step further
		for i := 0; i < t.step; i++ {
			_, _, err := buf.ReadRune()
			if err != nil {
				return nil, err
			}
		}
	}
	pos, err := t.rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	section := &Section{
		vocabSize: t.vocabSize,
		offset:    0,
		sentence:  make([]int, t.step),
	}
	for i := 0; i < t.batchSize; i++ {
		rne, _, err := buf.ReadRune()
		idx, err := t.runeToIdx(rne)
		if err != nil {
			return nil, err
		}
		section.sentence[i] = idx
	}
	// put the offset back where is was
	_, err = t.rs.Seek(pos, io.SeekStart)
	t.pass++
	return section, nil
}
