package lstm

import G "gorgonia.org/gorgonia"

// Forward pass as described here https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
func (l *lstm) fwd(inputVector, prevHidden, prevCell *G.Node) (hidden, cell *G.Node) {
	// Helper function for clarity
	set := func(ident, equation string) *G.Node {
		res, _ := l.parser.Parse(equation)
		l.parser.Set(ident, res)
		return res
	}

	l.parser.Set(`xₜ`, inputVector)
	l.parser.Set(`hₜ₋₁`, prevHidden)
	l.parser.Set(`cₜ₋₁`, prevCell)
	l.parser.Set(`fₜ`, σ(l, `Wf·xₜ+Uf·hₜ₋₁+Bf`)) // dot product made with ctrl+k . M
	l.parser.Set(`iₜ`, σ(l, `Wᵢ·xₜ+Uᵢ·hₜ₋₁+Bᵢ`))
	l.parser.Set(`oₜ`, σ(l, `Wₒ·xₜ+Uₒ·hₜ₋₁+Bₒ`))
	// ċₜis a vector of new candidates value
	l.parser.Set(`ĉₜ`, tanh(l, `Wc·xₜ+Uc·hₜ₋₁+Bc`)) // c made with ctrl+k c >
	ct := set(`cₜ`, `fₜ*cₜ₋₁+iₜ*ĉₜ`)
	l.parser.Set(`hc`, tanh(l, `cₜ`))
	ht, _ := l.parser.Parse(`oₜ*hc`)
	return ht, ct
}
