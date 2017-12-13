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

func σ(a *G.Node) *G.Node {
	return G.Must(G.Sigmoid(a))
}

func tanh(a *G.Node) *G.Node {
	return G.Must(G.Tanh(a))
}

func op(a *G.Node, operation rune, b *G.Node) *G.Node {
	switch operation {
	case '*':
		return G.Must(G.HadamardProd(a, b))
	case '·':
		return G.Must(G.Mul(a, b))
	case '+':
		return G.Must(G.Add(a, b))
	default:
		panic("Unknown operation")
	}
	return nil
}

func (m *lstm) step(cₜₜ, hₜₜ, xₜ *G.Node) (cₜ, hₜ *G.Node) {
	// fₜ= σ(Wf·[hₜ₋₁,xₜ] + bf)
	// fₜ= σ(Wf·hₜ₋₁+ Wf·xₜ+ bf)
	fₜ := σ(op(op(op(m.wf, '·', xₜ), '+', op(m.wf, '·', hₜₜ)), '+', m.bf))
	// iₜ= σ(Wi·[hₜ₋₁,xₜ] + bi)
	iₜ := σ(op(op(op(m.wi, '·', xₜ), '+', op(m.wi, '·', hₜₜ)), '+', m.bi))
	// Ĉₜ= tanh(Wc·[hₜ₋₁,xₜ] + bc)
	Ĉₜ := tanh(op(op(op(m.wc, '·', xₜ), '+', op(m.wc, '·', hₜₜ)), '+', m.bc))
	// Cₜ= fₜ* Cₜ₋₁+ iₜ* Ĉₜ
	cₜ = op(op(fₜ, '*', cₜₜ), '+', op(iₜ, '*', Ĉₜ))
	// oₜ= σ(Wo·[hₜ₋₁,xₜ] + bo)
	oₜ := σ(op(op(op(m.wo, '·', xₜ), '+', op(m.wo, '·', hₜₜ)), '+', m.bo))
	// hₜ= oₜ* tanh(Cₜ)
	hₜ = op(oₜ, '*', tanh(cₜ))

	return
}
