package parser_test

import (
	"testing"

	"github.com/owulveryck/charRNN/parser"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func σ(a *G.Node) *G.Node {
	return G.Must(G.Sigmoid(a))
}

func TestParse(t *testing.T) {
	g := G.NewGraph()
	wfT := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 1, 1, 1}))
	wf := G.NewMatrix(g, tensor.Float32, G.WithName("wf"), G.WithShape(2, 2), G.WithValue(wfT))
	htprevT := tensor.New(tensor.WithBacking([]float32{1, 1}), tensor.WithShape(2))
	htprev := G.NewVector(g, tensor.Float32, G.WithName("ht-1"), G.WithShape(2), G.WithValue(htprevT))
	xtT := tensor.New(tensor.WithBacking([]float32{1, 1}), tensor.WithShape(2))
	xt := G.NewVector(g, tensor.Float32, G.WithName("xt"), G.WithShape(2), G.WithValue(xtT))
	bfT := tensor.New(tensor.WithBacking([]float32{1, 1}), tensor.WithShape(2))
	bf := G.NewVector(g, tensor.Float32, G.WithName("bf"), G.WithShape(2), G.WithValue(bfT))

	p := parser.NewParser(g)
	p.Set(`Wf`, wf)
	p.Set(`hₜ₋₁`, htprev)
	p.Set(`xₜ`, xt)
	p.Set(`bf`, bf)
	result, err := p.Parse(`1*Wf·hₜ₋₁+ Wf·xₜ+ bf`)
	if err != nil {
		t.Fatal(err)
	}
	//result, _ := p.Parse(`Wf·hₜ₋₁`)
	//result = σ(result)
	machine := G.NewLispMachine(g, G.ExecuteFwdOnly())
	//machine := G.NewTapeMachine(g)
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	res := result.Value().Data().([]float32)
	if len(res) != 2 {
		t.Fail()
	}
	if res[0] != 5 && res[1] != 5 {
		t.Fail()
	}
}
