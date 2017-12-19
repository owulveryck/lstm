package lstm

import (
	"fmt"
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestFwd(t *testing.T) {
	model := NewModel(2, 2, []int{2})
	inputT := tensor.New(tensor.WithBacking([]float32{1, 1}), tensor.WithShape(2))
	input := G.NewVector(model.g, tensor.Float32, G.WithName("input"), G.WithShape(2), G.WithValue(inputT))
	hiddenT := tensor.New(tensor.WithBacking([]float32{1, 1}), tensor.WithShape(2))
	hidden := G.NewVector(model.g, tensor.Float32, G.WithName("hidden"), G.WithShape(2), G.WithValue(hiddenT))
	cellT := tensor.New(tensor.WithBacking([]float32{1, 1}), tensor.WithShape(2))
	cell := G.NewVector(model.g, tensor.Float32, G.WithName("cell"), G.WithShape(2), G.WithValue(cellT))
	model.ls[0].fwd(input, hidden, cell)
	fmt.Println(model.g.ToDot())
}
