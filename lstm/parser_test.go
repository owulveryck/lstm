package lstm_test

import (
	"fmt"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/owulveryck/charRNN/lstm"
	G "gorgonia.org/gorgonia"
)

func σ(a *G.Node) *G.Node {
	return G.Must(G.Sigmoid(a))
}

// Ensure the parser can parse strings into Statement ASTs.
func TestParser_ParseStatement(t *testing.T) {
	g := G.NewGraph()
	wfT := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 1, 1, 1}))
	wf := G.NewMatrix(g, tensor.Float32, G.WithName("wf"), G.WithShape(2, 2), G.WithValue(wfT))
	htprevT := tensor.New(tensor.WithBacking([]float32{1, 1}))
	htprev := G.NewVector(g, tensor.Float32, G.WithName("wf"), G.WithShape(2), G.WithValue(htprevT))
	xtT := tensor.New(tensor.WithBacking([]float32{1, 1}))
	xt := G.NewVector(g, tensor.Float32, G.WithName("wf"), G.WithShape(2), G.WithValue(xtT))
	bfT := tensor.New(tensor.WithBacking([]float32{1, 1}))
	bf := G.NewVector(g, tensor.Float32, G.WithName("wf"), G.WithShape(2), G.WithValue(bfT))

	parser := lstm.NewParser()
	parser.Let(`Wf`, wf)
	parser.Let(`hₜ₋₁`, htprev)
	parser.Let(`xₜ`, xt)
	parser.Let(`bh`, bf)
	/*
		_, err := parser.Parse(`Wf·hₜ₋₁`)
		if err != nil {
			t.Fatal(err)
		}
	*/
	node, err := parser.Parse(`Wf·hₜ₋₁+ Wf·xₜ+ bf`)
	if err != nil {
		t.Fatal(err)
	}
	σ(node)
	// fₜ= σ(Wf·hₜ₋₁+ Wf·xₜ+ bf)
	fmt.Println(g.ToDot())
}
