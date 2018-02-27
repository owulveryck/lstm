package lstm

import (
	"io"

	G "gorgonia.org/gorgonia"
)

// IO is a structure that handles
type IO struct {
	g *G.ExprGraph
}

func (i *IO) writeCurrentInputVector(*G.Node) error {
	return nil
}
func (i *IO) getCurrentInputVector() (*G.Node, error) {
	return nil, nil
}

// Forward pass as described here https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
func (m *Model) fwd(tset IO, prevHidden, prevCell *G.Node) error {
	inputVector, err := tset.getCurrentInputVector()
	if err != nil {
		if err == io.EOF {
			return nil
		}
		return err
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
	return m.fwd(tset, ht, ct)
}
