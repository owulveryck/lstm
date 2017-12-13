package lstm

import G "gorgonia.org/gorgonia"

type lstm struct{}

func (m *lstm) step(c, h, x *G.Node) (*G.Node, *G.Node) {
	// fₜ= σ(Wf·[hₜ₋₁,xₜ] + bf)
	// iₜ= σ(Wi·[hₜ₋₁,xₜ] + bi)
	// Ĉₜ= tanh(Wc·[hₜ₋₁,xₜ] + bc)
	// Cₜ= fₜ* Cₜ₋₁+ iₜ* Ĉₜ
	// oₜ= σ(Wo·[hₜ₋₁,xₜ] + bo)
	// hₜ= oₜ*tanh(Cₜ)

	return nil, nil
}
