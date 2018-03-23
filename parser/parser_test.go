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
	//result, err := p.Parse(`σ(1*Wf·hₜ₋₁+ Wf·xₜ+ bf)`)
	type test struct {
		equation string
		expected []float32
	}
	for _, test := range []test{
		{
			`1*Wf·hₜ₋₁+ Wf·xₜ+ bf`,
			[]float32{5, 5},
		},
		{
			`σ(1*Wf·hₜ₋₁+ Wf·xₜ+ bf)`,
			[]float32{0.9933072, 0.9933072},
		},
		{
			`tanh(1*Wf·hₜ₋₁+ Wf·xₜ+ bf)`,
			[]float32{0.9999092, 0.9999092},
		},
	} {
		result, err := p.Parse(test.equation)
		if err != nil {
			t.Fatal(err)
		}
		machine := G.NewLispMachine(g, G.ExecuteFwdOnly())
		if err := machine.RunAll(); err != nil {
			t.Fatal(err)
		}
		res := result.Value().Data().([]float32)
		if len(res) != 2 {
			t.Fail()
		}
		if res[0] != test.expected[0] || res[1] != test.expected[1] {
			t.Fail()
		}
	}
}
