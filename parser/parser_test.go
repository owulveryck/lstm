package parser_test

import (
	"fmt"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/owulveryck/charRNN/parser"
	G "gorgonia.org/gorgonia"
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

	p := parser.NewParser()
	p.Set(`Wf`, wf)
	p.Set(`hₜ₋₁`, htprev)
	p.Set(`xₜ`, xt)
	p.Set(`bh`, bf)
	_, err := p.Parse(`Wf·hₜ₋₁+ Wf·xₜ+ bf`)
	if err != nil {
		//		t.Fatal(err)
	}
	fmt.Println(g.ToDot())
	//σ(node)
	// fₜ= σ(Wf·hₜ₋₁+ Wf·xₜ+ bf)
	fmt.Println(g.ToDot())

}
