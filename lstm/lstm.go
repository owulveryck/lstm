package lstm

import (
	"bufio"
	"io"

	G "gorgonia.org/gorgonia"
)

// IO is a structure that handles a batch training set
type IO struct {
	RS io.ReadSeeker
	WS io.ReadWriteSeeker
	*Reader
}

type Reader struct {
	r   io.Reader
	buf *bufio.Reader
}

func (i *IO) WriteVector(*G.Node) error {
	return nil
}

func NewReader(r io.Reader) *Reader {
	return &Reader{
		r:   r,
		buf: bufio.NewReader(r),
	}
}
func (r *Reader) ReadVector() ([]float32, error) {
	char, _, err := r.buf.ReadRune()
	if err != nil {
		return nil, err
	}
	return oneOfK(char)
}

func oneOfK(r rune) ([]float32, error) {
	return nil, nil
}
func (i *IO) getLastOutputVector() (*G.Node, error) {
	return nil, nil
}

// Forward pass as described here https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
func (m *Model) fwd(tset IO, prevHidden, prevCell *G.Node, currentGeneratedIndex, maxGeneratedIndex int) error {
	inputVector, err := tset.ReadVector()
	switch {
	case err != nil && err != io.EOF:
		return err
	case err == io.EOF && currentGeneratedIndex == maxGeneratedIndex:
		return nil
	case err == io.EOF && currentGeneratedIndex < maxGeneratedIndex:
		inputVector, err = tset.getLastOutputVector()
		if err != nil {
			return err
		}
	}
	// Helper function for clarity
	set := func(ident, equation string) *G.Node {
		res, _ := m.parser.Parse(equation)
		m.parser.Set(ident, res)
		return res
	}

	m.parser.Set(`xₜ`, inputVector)
	m.parser.Set(`hₜ₋₁`, prevHidden)
	m.parser.Set(`cₜ₋₁`, prevCell)
	set(`iₜ`, `σ(Wᵢ·xₜ+Uᵢ·hₜ₋₁+Bᵢ)`)
	set(`fₜ`, `σ(Wf·xₜ+Uf·hₜ₋₁+Bf)`) // dot product made with ctrl+k . M
	set(`oₜ`, `σ(Wₒ·xₜ+Uₒ·hₜ₋₁+Bₒ)`)
	// ċₜis a vector of new candidates value
	set(`ĉₜ`, `tanh(Wc·xₜ+Uc·hₜ₋₁+Bc)`) // c made with ctrl+k c >
	ct := set(`cₜ`, `fₜ*cₜ₋₁+iₜ*ĉₜ`)
	set(`hc`, `tanh(cₜ)`)
	ht, _ := m.parser.Parse(`oₜ*hc`)
	tset.writeCurrentInputVector(ht)
	return m.fwd(tset, ht, ct, currentGeneratedIndex+1, maxGeneratedIndex)
}
