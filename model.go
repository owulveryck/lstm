package lstm

import (
	"github.com/owulveryck/lstm/parser"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Model holds the tensor of the model
type Model struct {
	wiT    tensor.Tensor
	uiT    tensor.Tensor
	biasIT tensor.Tensor

	wfT    tensor.Tensor
	ufT    tensor.Tensor
	biasFT tensor.Tensor

	woT    tensor.Tensor
	uoT    tensor.Tensor
	biasOT tensor.Tensor

	wcT    tensor.Tensor
	ucT    tensor.Tensor
	biasCT tensor.Tensor

	wyT    tensor.Tensor
	biasYT tensor.Tensor

	inputSize  int
	outputSize int
	hiddenSize int
}

// lstm represent a single cell of the RNN
// each LSTM owns its own ExprGraph
type lstm struct {
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

func (m *Model) newLSTM(hiddenT, cellT tensor.Tensor) *lstm {
	lstm := new(lstm)
	g := G.NewGraph()
	lstm.g = g
	p := parser.NewParser(g)
	lstm.parser = p
	lstm.hiddenSize = m.hiddenSize
	lstm.inputSize = m.inputSize
	lstm.outputSize = m.outputSize

	prevSize := m.inputSize
	hiddenSize := m.hiddenSize
	outputSize := m.outputSize

	// input gate weights
	lstm.wi = G.NewMatrix(g, tensor.Float32, G.WithName("Wᵢ"), G.WithShape(hiddenSize, prevSize), G.WithValue(m.wiT))
	lstm.ui = G.NewMatrix(g, tensor.Float32, G.WithName("Uᵢ"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(m.uiT))
	lstm.biasI = G.NewVector(g, tensor.Float32, G.WithName("Bᵢ"), G.WithShape(hiddenSize), G.WithValue(m.biasIT))
	p.Set(`Wᵢ`, lstm.wi)
	p.Set(`Uᵢ`, lstm.ui)
	p.Set(`Bᵢ`, lstm.biasI)

	// output gate weights
	lstm.wo = G.NewMatrix(g, tensor.Float32, G.WithName("Wₒ"), G.WithShape(hiddenSize, prevSize), G.WithValue(m.woT))
	lstm.uo = G.NewMatrix(g, tensor.Float32, G.WithName("Uₒ"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(m.uoT))
	lstm.biasO = G.NewVector(g, tensor.Float32, G.WithName("Bₒ"), G.WithShape(hiddenSize), G.WithValue(m.biasOT))
	p.Set(`Wₒ`, lstm.wo)
	p.Set(`Uₒ`, lstm.uo)
	p.Set(`Bₒ`, lstm.biasO)

	// forget gate weights
	lstm.wf = G.NewMatrix(g, tensor.Float32, G.WithName("Wf"), G.WithShape(hiddenSize, prevSize), G.WithValue(m.wfT))
	lstm.uf = G.NewMatrix(g, tensor.Float32, G.WithName("Uf"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(m.ufT))
	lstm.biasF = G.NewVector(g, tensor.Float32, G.WithName("Bf"), G.WithShape(hiddenSize), G.WithValue(m.biasFT))
	p.Set(`Wf`, lstm.wf)
	p.Set(`Uf`, lstm.uf)
	p.Set(`Bf`, lstm.biasF)

	// cell write
	lstm.wc = G.NewMatrix(g, tensor.Float32, G.WithName("Wc"), G.WithShape(hiddenSize, prevSize), G.WithValue(m.wcT))
	lstm.uc = G.NewMatrix(g, tensor.Float32, G.WithName("Uc"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(m.ucT))
	lstm.biasC = G.NewVector(g, tensor.Float32, G.WithName("bc"), G.WithShape(hiddenSize), G.WithValue(m.biasCT))
	p.Set(`Wc`, lstm.wc)
	p.Set(`Uc`, lstm.uc)
	p.Set(`Bc`, lstm.biasC)

	// Output vector
	lstm.wy = G.NewMatrix(g, tensor.Float32, G.WithName("Wy"), G.WithShape(outputSize, hiddenSize), G.WithValue(m.wyT))
	lstm.biasY = G.NewVector(g, tensor.Float32, G.WithName("by"), G.WithShape(outputSize), G.WithValue(m.biasYT))
	p.Set(`Wy`, lstm.wy)
	p.Set(`By`, lstm.biasY)

	// this is to simulate a default "previous" state
	lstm.prevHidden = G.NewVector(g, tensor.Float32, G.WithName("hₜ₋₁"), G.WithShape(hiddenSize), G.WithValue(hiddenT))
	lstm.prevCell = G.NewVector(g, tensor.Float32, G.WithName("Cₜ₋₁"), G.WithShape(hiddenSize), G.WithValue(cellT))

	return lstm
}

func newModelFromBackends(back *backends) *Model {
	m := new(Model)
	m.hiddenSize = back.HiddenSize
	m.inputSize = back.InputSize
	m.outputSize = back.OutputSize

	prevSize := back.InputSize
	hiddenSize := back.HiddenSize
	outputSize := back.OutputSize

	// input gate weights
	m.wiT = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wi))
	m.uiT = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Ui))
	m.biasIT = tensor.New(tensor.WithBacking(back.BiasI), tensor.WithShape(hiddenSize))

	// output gate weights
	m.woT = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wo))
	m.uoT = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Uo))
	m.biasOT = tensor.New(tensor.WithBacking(back.BiasO), tensor.WithShape(hiddenSize))

	// forget gate weights
	m.wfT = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wf))
	m.ufT = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Uf))
	m.biasFT = tensor.New(tensor.WithBacking(back.BiasF), tensor.WithShape(hiddenSize))

	// cell write
	m.wcT = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wc))
	m.ucT = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Uc))
	m.biasCT = tensor.New(tensor.WithBacking(back.BiasC), tensor.WithShape(hiddenSize))

	// Output vector
	m.wyT = tensor.New(tensor.WithShape(outputSize, hiddenSize), tensor.WithBacking(back.Wy))
	m.biasYT = tensor.New(tensor.WithBacking(back.BiasY), tensor.WithShape(outputSize))
	return m
}

// NewModel creates a new model
func NewModel(inputSize, outputSize int, hiddenSize int) *Model {
	return newModelFromBackends(initBackends(inputSize, outputSize, hiddenSize))
}
