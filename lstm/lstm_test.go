package lstm

import (
	"log"
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

// https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
var (
	wcTest = []float32{0.45, 0.25}
	wiTest = []float32{0.95, 0.8}
	wfTest = []float32{0.7, 0.45}
	woTest = []float32{0.6, 0.4}
	ucTest = []float32{0.15}
	uiTest = []float32{0.8}
	ufTest = []float32{0.1}
	uoTest = []float32{0.25}
	bcTest = []float32{0.2}
	biTest = []float32{0.65}
	bfTest = []float32{0.15}
	boTest = []float32{0.1}
	x0     = []float32{1, 2}
	y0     = []float32{0.5}
	x1     = []float32{0.5, 3}
	y1     = []float32{1.25}
	c0     = []float32{0.78572}
	h0     = []float32{0.53631}
	c1     = []float32{1.5176}
	h1     = []float32{0.77197}
)

func exampleBackend(inputSize, outputSize int, hiddenSizes []int) *backends {
	var back backends
	back.InputSize = inputSize
	back.OutputSize = outputSize
	back.HiddenSizes = hiddenSizes
	numlstms := len(hiddenSizes)
	back.wi = make([][]float32, numlstms)
	back.ui = make([][]float32, numlstms)
	back.BiasI = make([][]float32, numlstms)
	back.wf = make([][]float32, numlstms)
	back.uf = make([][]float32, numlstms)
	back.BiasF = make([][]float32, numlstms)
	back.wo = make([][]float32, numlstms)
	back.uo = make([][]float32, numlstms)
	back.BiasO = make([][]float32, numlstms)
	back.wc = make([][]float32, numlstms)
	back.uc = make([][]float32, numlstms)
	back.BiasC = make([][]float32, numlstms)
	for depth := 0; depth < numlstms; depth++ {
		back.wi[depth] = wiTest
		back.ui[depth] = uiTest
		back.BiasI[depth] = biTest
		back.wo[depth] = woTest
		back.uo[depth] = uoTest
		back.BiasO[depth] = boTest
		back.wf[depth] = wfTest
		back.uf[depth] = ufTest
		back.BiasF[depth] = bfTest
		back.wc[depth] = wcTest
		back.uc[depth] = ucTest
		back.BiasC[depth] = bcTest
	}
	back.Whd = makeVectorIdentity(back.OutputSize, hiddenSizes[len(hiddenSizes)-1])
	back.BiasD = make([]float32, outputSize)
	return &back
}

func identityBackend(inputSize, outputSize int, hiddenSizes []int) *backends {
	var back backends
	back.InputSize = inputSize
	back.OutputSize = outputSize
	back.HiddenSizes = hiddenSizes
	numlstms := len(hiddenSizes)
	back.wi = make([][]float32, numlstms)
	back.ui = make([][]float32, numlstms)
	back.BiasI = make([][]float32, numlstms)
	back.wf = make([][]float32, numlstms)
	back.uf = make([][]float32, numlstms)
	back.BiasF = make([][]float32, numlstms)
	back.wo = make([][]float32, numlstms)
	back.uo = make([][]float32, numlstms)
	back.BiasO = make([][]float32, numlstms)
	back.wc = make([][]float32, numlstms)
	back.uc = make([][]float32, numlstms)
	back.BiasC = make([][]float32, numlstms)
	for depth := 0; depth < numlstms; depth++ {
		prevSize := inputSize
		if depth > 0 {
			prevSize = hiddenSizes[depth-1]
		}
		hiddenSize := hiddenSizes[depth]
		back.wi[depth] = makeVectorIdentity(hiddenSize, prevSize)
		back.ui[depth] = makeVectorIdentity(hiddenSize, hiddenSize)
		back.BiasI[depth] = make([]float32, hiddenSize)
		back.wo[depth] = makeVectorIdentity(hiddenSize, prevSize)
		back.uo[depth] = makeVectorIdentity(hiddenSize, hiddenSize)
		back.BiasO[depth] = make([]float32, hiddenSize)
		back.wf[depth] = makeVectorIdentity(hiddenSize, prevSize)
		back.uf[depth] = makeVectorIdentity(hiddenSize, hiddenSize)
		back.BiasF[depth] = make([]float32, hiddenSize)
		back.wc[depth] = makeVectorIdentity(hiddenSize, prevSize)
		back.uc[depth] = makeVectorIdentity(hiddenSize, hiddenSize)
		back.BiasC[depth] = make([]float32, hiddenSize)
	}
	back.Whd = makeVectorIdentity(back.OutputSize, hiddenSizes[len(hiddenSizes)-1])
	back.BiasD = make([]float32, outputSize)
	return &back
}

func TestBackpass(t *testing.T) {
	model := newModelFromBackends(exampleBackend(2, 1, []int{1}))
	inputT := tensor.New(tensor.WithBacking(x0), tensor.WithShape(2))
	input := G.NewVector(model.g, tensor.Float32, G.WithName("xₜ"), G.WithShape(2), G.WithValue(inputT))
	hiddenT := tensor.New(tensor.WithBacking([]float32{0}), tensor.WithShape(1))
	hidden := G.NewVector(model.g, tensor.Float32, G.WithName("hₜ₋₁"), G.WithShape(1), G.WithValue(hiddenT))
	cellT := tensor.New(tensor.WithBacking([]float32{0}), tensor.WithShape(1))
	cprev := G.NewVector(model.g, tensor.Float32, G.WithName("cₜ₋₁"), G.WithShape(1), G.WithValue(cellT))
	c, h := model.ls[0].fwd(input, hidden, cprev)
	//fmt.Println(model.g.ToDot())
	machine := G.NewLispMachine(model.g, G.ExecuteFwdOnly())
	//machine := G.NewTapeMachine(g)
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	hv := h.Value().Data().([]float32)
	cv := c.Value().Data().([]float32)
	log.Println(hv)
	log.Println(cv)
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
