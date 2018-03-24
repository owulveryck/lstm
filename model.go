package lstm

import (
	"github.com/owulveryck/lstm/parser"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Model holds the structure of the LSTM model
type Model struct {
	g     *G.ExprGraph
	wi    *G.Node
	ui    *G.Node
	biasI *G.Node

	wf    *G.Node
	uf    *G.Node
	biasF *G.Node

	wo    *G.Node
	uo    *G.Node
	biasO *G.Node

	wc    *G.Node
	uc    *G.Node
	biasC *G.Node

	wy     *G.Node
	biasY  *G.Node
	parser *parser.Parser

	inputSize  int
	outputSize int
	hiddenSize int
	//inputVector *G.Node
	prevHidden *G.Node
	prevCell   *G.Node
}

func newModelFromBackends(back *backends) *Model {
	m := new(Model)
	g := G.NewGraph()
	m.hiddenSize = back.HiddenSize
	m.inputSize = back.InputSize
	m.outputSize = back.OutputSize

	prevSize := back.InputSize
	hiddenSize := back.HiddenSize
	outputSize := back.OutputSize
	p := parser.NewParser(g)
	m.parser = p

	// input gate weights
	wiT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wi))
	uiT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Ui))
	//biasIT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
	biasIT := tensor.New(tensor.WithBacking(back.BiasI), tensor.WithShape(hiddenSize))

	m.wi = G.NewMatrix(g, tensor.Float32, G.WithName("Wᵢ"), G.WithShape(hiddenSize, prevSize), G.WithValue(wiT))
	m.ui = G.NewMatrix(g, tensor.Float32, G.WithName("Uᵢ"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(uiT))
	m.biasI = G.NewVector(g, tensor.Float32, G.WithName("Bᵢ"), G.WithShape(hiddenSize), G.WithValue(biasIT))
	p.Set(`Wᵢ`, m.wi)
	p.Set(`Uᵢ`, m.ui)
	p.Set(`Bᵢ`, m.biasI)

	// output gate weights

	woT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wo))
	uoT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Uo))
	biasOT := tensor.New(tensor.WithBacking(back.BiasO), tensor.WithShape(hiddenSize))

	m.wo = G.NewMatrix(g, tensor.Float32, G.WithName("Wₒ"), G.WithShape(hiddenSize, prevSize), G.WithValue(woT))
	m.uo = G.NewMatrix(g, tensor.Float32, G.WithName("Uₒ"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(uoT))
	m.biasO = G.NewVector(g, tensor.Float32, G.WithName("Bₒ"), G.WithShape(hiddenSize), G.WithValue(biasOT))
	p.Set(`Wₒ`, m.wo)
	p.Set(`Uₒ`, m.uo)
	p.Set(`Bₒ`, m.biasO)

	// forget gate weights

	wfT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wf))
	ufT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Uf))
	biasFT := tensor.New(tensor.WithBacking(back.BiasF), tensor.WithShape(hiddenSize))

	m.wf = G.NewMatrix(g, tensor.Float32, G.WithName("Wf"), G.WithShape(hiddenSize, prevSize), G.WithValue(wfT))
	m.uf = G.NewMatrix(g, tensor.Float32, G.WithName("Uf"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(ufT))
	m.biasF = G.NewVector(g, tensor.Float32, G.WithName("Bf"), G.WithShape(hiddenSize), G.WithValue(biasFT))
	p.Set(`Wf`, m.wf)
	p.Set(`Uf`, m.uf)
	p.Set(`Bf`, m.biasF)

	// cell write

	wcT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wc))
	ucT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Uc))
	biasCT := tensor.New(tensor.WithBacking(back.BiasC), tensor.WithShape(hiddenSize))

	m.wc = G.NewMatrix(g, tensor.Float32, G.WithName("Wc"), G.WithShape(hiddenSize, prevSize), G.WithValue(wcT))
	m.uc = G.NewMatrix(g, tensor.Float32, G.WithName("Uc"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(ucT))
	m.biasC = G.NewVector(g, tensor.Float32, G.WithName("bc"), G.WithShape(hiddenSize), G.WithValue(biasCT))
	p.Set(`Wc`, m.wc)
	p.Set(`Uc`, m.uc)
	p.Set(`Bc`, m.biasC)

	// Output vector
	wyT := tensor.New(tensor.WithShape(outputSize, hiddenSize), tensor.WithBacking(back.Wy))
	biasYT := tensor.New(tensor.WithBacking(back.BiasY), tensor.WithShape(outputSize))

	m.wy = G.NewMatrix(g, tensor.Float32, G.WithName("Wy"), G.WithShape(outputSize, hiddenSize), G.WithValue(wyT))
	m.biasY = G.NewVector(g, tensor.Float32, G.WithName("by"), G.WithShape(outputSize), G.WithValue(biasYT))
	p.Set(`Wy`, m.wy)
	p.Set(`By`, m.biasY)

	// this is to simulate a default "previous" state
	hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
	cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
	m.prevHidden = G.NewVector(g, tensor.Float32, G.WithName("hₜ₋₁"), G.WithShape(hiddenSize), G.WithValue(hiddenT))
	m.prevCell = G.NewVector(g, tensor.Float32, G.WithName("Cₜ₋₁"), G.WithShape(hiddenSize), G.WithValue(cellT))

	// these are to simulate a previous state
	//dummyInputVec := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(back.InputSize)) // zeroes
	//m.inputVector = G.NewVector(g, tensor.Float32, G.WithName("xₜ"), G.WithShape(back.InputSize), G.WithValue(dummyInputVec))

	m.g = g
	return m
}

// NewModel creates a new model
func NewModel(inputSize, outputSize int, hiddenSize int) *Model {
	return newModelFromBackends(initBackends(inputSize, outputSize, hiddenSize))
}
