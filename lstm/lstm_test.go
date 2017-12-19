package lstm

import (
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func makeVectorIdentity(x, y int) []float32 {
	vec := make([]float32, x*y)
	for i := range vec {
		vec[i] = 1
	}
	return vec
}

func identityBackend(inputSize, outputSize int, hiddenSizes []int) *backends {
	var back backends
	back.InputSize = inputSize
	back.OutputSize = outputSize
	back.HiddenSizes = hiddenSizes
	numlstms := len(hiddenSizes)
	back.Wix = make([][]float32, numlstms)
	back.Wih = make([][]float32, numlstms)
	back.BiasI = make([][]float32, numlstms)
	back.Wfx = make([][]float32, numlstms)
	back.Wfh = make([][]float32, numlstms)
	back.BiasF = make([][]float32, numlstms)
	back.Wox = make([][]float32, numlstms)
	back.Woh = make([][]float32, numlstms)
	back.BiasO = make([][]float32, numlstms)
	back.Wcx = make([][]float32, numlstms)
	back.Wch = make([][]float32, numlstms)
	back.BiasC = make([][]float32, numlstms)
	for depth := 0; depth < numlstms; depth++ {
		prevSize := inputSize
		if depth > 0 {
			prevSize = hiddenSizes[depth-1]
		}
		hiddenSize := hiddenSizes[depth]
		back.Wix[depth] = makeVectorIdentity(hiddenSize, prevSize)
		back.Wih[depth] = makeVectorIdentity(hiddenSize, hiddenSize)
		back.BiasI[depth] = make([]float32, hiddenSize)
		back.Wox[depth] = makeVectorIdentity(hiddenSize, prevSize)
		back.Woh[depth] = makeVectorIdentity(hiddenSize, hiddenSize)
		back.BiasO[depth] = make([]float32, hiddenSize)
		back.Wfx[depth] = makeVectorIdentity(hiddenSize, prevSize)
		back.Wfh[depth] = makeVectorIdentity(hiddenSize, hiddenSize)
		back.BiasF[depth] = make([]float32, hiddenSize)
		back.Wcx[depth] = makeVectorIdentity(hiddenSize, prevSize)
		back.Wch[depth] = makeVectorIdentity(hiddenSize, hiddenSize)
		back.BiasC[depth] = make([]float32, hiddenSize)
	}
	back.Whd = makeVectorIdentity(back.OutputSize, hiddenSizes[len(hiddenSizes)-1])
	back.BiasD = make([]float32, outputSize)
	return &back
}

func TestFwd(t *testing.T) {
	model := newModelFromBackends(identityBackend(2, 2, []int{2}))
	inputT := tensor.New(tensor.WithBacking([]float32{1, 1}), tensor.WithShape(2))
	input := G.NewVector(model.g, tensor.Float32, G.WithName("xₜ"), G.WithShape(2), G.WithValue(inputT))
	hiddenT := tensor.New(tensor.WithBacking([]float32{1, 1}), tensor.WithShape(2))
	hidden := G.NewVector(model.g, tensor.Float32, G.WithName("hₜ₋₁"), G.WithShape(2), G.WithValue(hiddenT))
	cellT := tensor.New(tensor.WithBacking([]float32{1, 1}), tensor.WithShape(2))
	cprev := G.NewVector(model.g, tensor.Float32, G.WithName("cₜ₋₁"), G.WithShape(2), G.WithValue(cellT))
	c, h := model.ls[0].fwd(input, hidden, cprev)
	machine := G.NewLispMachine(model.g, G.ExecuteFwdOnly())
	//machine := G.NewTapeMachine(g)
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	hv := h.Value().Data().([]float32)
	cv := c.Value().Data().([]float32)
	if len(hv) != 2 {
		t.Fail()
	}
	if hv[0] != hv[1] {
		t.Fail()
	}
	if cv[0] != cv[1] {
		t.Fail()
	}
	if hv[0] != 1.7299098 {
		t.Fail()
	}
	if cv[0] != 0.8271083 {
		t.Fail()
	}
	//fmt.Println(model.g.ToDot())
}
