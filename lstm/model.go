package lstm

import G "gorgonia.org/gorgonia"

type lstm struct {
	wf *G.Node
	bf *G.Node
	wi *G.Node
	bi *G.Node
	wo *G.Node
	bo *G.Node
	wc *G.Node
	bc *G.Node
	g  G.ExprGraph
}

func (m *lstm) step(cₜₜ, hₜₜ, xₜ *G.Node) (cₜ, hₜ *G.Node) {
	// fₜ= σ(Wf·[hₜ₋₁,xₜ] + bf)
	// iₜ= σ(Wi·[hₜ₋₁,xₜ] + bi)
	// Ĉₜ= tanh(Wc·[hₜ₋₁,xₜ] + bc)
	// Cₜ= fₜ* Cₜ₋₁+ iₜ* Ĉₜ
	// oₜ= σ(Wo·[hₜ₋₁,xₜ] + bo)
	// hₜ= oₜ* tanh(Cₜ)

	return nil, nil
}
