package lstm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
func TestNewCell_forward_pass(t *testing.T) {
	vectorSize := 3
	hiddenSize := 2
	lstm := NewLSTM(vectorSize, hiddenSize)
	initWeights(lstm)

	x := gorgonia.NewVector(lstm.G, float, gorgonia.WithName("xₜ"),
		gorgonia.WithShape(vectorSize))
	hPrev := gorgonia.NewVector(lstm.G, float, gorgonia.WithName("hₜ₋₁"),
		gorgonia.WithShape(hiddenSize))
	cPrev := gorgonia.NewVector(lstm.G, float, gorgonia.WithName("cₜ₋₁"),
		gorgonia.WithShape(hiddenSize))
	h, c := lstm.NewCell(x, hPrev, cPrev)
	xT := tensor.NewDense(float, []int{vectorSize}, tensor.WithBacking([]float64{1, 2, 3}))
	hPrevT := tensor.NewDense(float, []int{hiddenSize}, tensor.WithBacking([]float64{1, 2}))
	cPrevT := tensor.NewDense(float, []int{hiddenSize}, tensor.WithBacking([]float64{1, 2}))
	gorgonia.Let(x, xT)
	gorgonia.Let(cPrev, cPrevT)
	gorgonia.Let(hPrev, hPrevT)
	vm := gorgonia.NewTapeMachine(lstm.G)
	err := vm.RunAll()
	if err != nil {
		t.Fatal(err)
	}
	vm.Close()
	assert.InDeltaSlice(t, []float64{0.960770898451274, 0.9929241586741123}, h.Value().Data().([]float64), 1e-10, "bad h value")
	assert.InDeltaSlice(t, []float64{0.9618409812007414, 0.9944934973187705}, c.Value().Data().([]float64), 1e-10, "bad c value")
	h2, c2 := lstm.NewCell(x, h, c)
	vm = gorgonia.NewTapeMachine(lstm.G)
	err = vm.RunAll()
	if err != nil {
		t.Fatal(err)
	}
	vm.Close()
	assert.InDeltaSlice(t, []float64{0.9916307952465716, 0.9956302280119376}, h2.Value().Data().([]float64), 1e-10, "bad h value")
	assert.InDeltaSlice(t, []float64{0.9940468069105712, 0.9990725394785818}, c2.Value().Data().([]float64), 1e-10, "bad c value")
}

func TestNewCell_forward_pass_composite(t *testing.T) {
	vectorSize := 3
	hiddenSize := 2
	lstm := NewLSTM(vectorSize, hiddenSize)
	initWeights(lstm)

	x := gorgonia.NewVector(lstm.G, float, gorgonia.WithName("xₜ"),
		gorgonia.WithShape(vectorSize))
	hPrev := gorgonia.NewVector(lstm.G, float, gorgonia.WithName("hₜ₋₁"),
		gorgonia.WithShape(hiddenSize))
	cPrev := gorgonia.NewVector(lstm.G, float, gorgonia.WithName("cₜ₋₁"),
		gorgonia.WithShape(hiddenSize))
	h, c := lstm.NewCell(x, hPrev, cPrev)
	xT := tensor.NewDense(float, []int{vectorSize}, tensor.WithBacking([]float64{1, 2, 3}))
	hPrevT := tensor.NewDense(float, []int{hiddenSize}, tensor.WithBacking([]float64{1, 2}))
	cPrevT := tensor.NewDense(float, []int{hiddenSize}, tensor.WithBacking([]float64{1, 2}))
	gorgonia.Let(x, xT)
	gorgonia.Let(cPrev, cPrevT)
	gorgonia.Let(hPrev, hPrevT)
	h2, c2 := lstm.NewCell(x, h, c)
	vm := gorgonia.NewTapeMachine(lstm.G)
	err := vm.RunAll()
	if err != nil {
		t.Fatal(err)
	}
	vm.Close()
	assert.InDeltaSlice(t, []float64{0.9916307952465716, 0.9956302280119376}, h2.Value().Data().([]float64), 1e-10, "bad h value")
	assert.InDeltaSlice(t, []float64{0.9940468069105712, 0.9990725394785818}, c2.Value().Data().([]float64), 1e-10, "bad c value")
}

func initWeights(l *LSTM) {
	wiT := tensor.NewDense(float, []int{l.HiddenSize, l.VectorSize}, tensor.WithBacking([]float64{0.45, 0.25, 0.45, 0.25, 0.45, 0.25}))
	wfT := tensor.NewDense(float, []int{l.HiddenSize, l.VectorSize}, tensor.WithBacking([]float64{0.55, 0.35, 0.55, 0.35, 0.55, 0.35}))
	wcT := tensor.NewDense(float, []int{l.HiddenSize, l.VectorSize}, tensor.WithBacking([]float64{0.65, 0.45, 0.65, 0.45, 0.65, 0.45}))
	woT := tensor.NewDense(float, []int{l.HiddenSize, l.VectorSize}, tensor.WithBacking([]float64{0.75, 0.55, 0.75, 0.55, 0.75, 0.55}))
	uiT := tensor.NewDense(float, []int{l.HiddenSize, l.HiddenSize}, tensor.WithBacking([]float64{0.4, 0.45, 0.4, 0.45}))
	ufT := tensor.NewDense(float, []int{l.HiddenSize, l.HiddenSize}, tensor.WithBacking([]float64{0.5, 0.55, 0.5, 0.55}))
	ucT := tensor.NewDense(float, []int{l.HiddenSize, l.HiddenSize}, tensor.WithBacking([]float64{0.6, 0.65, 0.6, 0.65}))
	uoT := tensor.NewDense(float, []int{l.HiddenSize, l.HiddenSize}, tensor.WithBacking([]float64{0.7, 0.75, 0.7, 0.75}))
	biT := tensor.NewDense(float, []int{l.HiddenSize}, tensor.WithBacking([]float64{0.2, 0.25}))
	bfT := tensor.NewDense(float, []int{l.HiddenSize}, tensor.WithBacking([]float64{0.3, 0.35}))
	bcT := tensor.NewDense(float, []int{l.HiddenSize}, tensor.WithBacking([]float64{0.4, 0.45}))
	boT := tensor.NewDense(float, []int{l.HiddenSize}, tensor.WithBacking([]float64{0.5, 0.55}))
	gorgonia.Let(l.Wi, wiT)
	gorgonia.Let(l.Wf, wfT)
	gorgonia.Let(l.Wc, wcT)
	gorgonia.Let(l.Wo, woT)
	gorgonia.Let(l.Ui, uiT)
	gorgonia.Let(l.Uf, ufT)
	gorgonia.Let(l.Uc, ucT)
	gorgonia.Let(l.Uo, uoT)
	gorgonia.Let(l.Bi, biT)
	gorgonia.Let(l.Bf, bfT)
	gorgonia.Let(l.Bc, bcT)
	gorgonia.Let(l.Bo, boT)
}
