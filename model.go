package lstm

import (
	"github.com/owulveryck/lstm/parser"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Model holds the tensor of the model
type Model struct {
	wi    []float32
	ui    []float32
	biasI []float32

	wf    []float32
	uf    []float32
	biasF []float32

	wo    []float32
	uo    []float32
	biasO []float32

	wc    []float32
	uc    []float32
	biasC []float32

	wy    []float32
	biasY []float32

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
	outputs    G.Nodes
}

func (m *Model) newLSTM(hiddenT, cellT tensor.Tensor) *lstm {
	lstm := new(lstm)
	g := G.NewGraph()
	lstm.g = g
	p := parser.NewParser(g)
	lstm.outputs = make(G.Nodes, 0)
	lstm.parser = p
	lstm.hiddenSize = m.hiddenSize
	lstm.inputSize = m.inputSize
	lstm.outputSize = m.outputSize

	prevSize := m.inputSize
	hiddenSize := m.hiddenSize
	outputSize := m.outputSize

	// Create the tensor first
	// input gate weights
	wiT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(m.wi))
	uiT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(m.ui))
	biasIT := tensor.New(tensor.WithBacking(m.biasI), tensor.WithShape(hiddenSize))

	// output gate weights
	woT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(m.wo))
	uoT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(m.uo))
	biasOT := tensor.New(tensor.WithBacking(m.biasO), tensor.WithShape(hiddenSize))

	// forget gate weights
	wfT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(m.wf))
	ufT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(m.uf))
	biasFT := tensor.New(tensor.WithBacking(m.biasF), tensor.WithShape(hiddenSize))

	// cell write
	wcT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(m.wc))
	ucT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(m.uc))
	biasCT := tensor.New(tensor.WithBacking(m.biasC), tensor.WithShape(hiddenSize))

	// Output vector
	wyT := tensor.New(tensor.WithShape(outputSize, hiddenSize), tensor.WithBacking(m.wy))
	biasYT := tensor.New(tensor.WithBacking(m.biasY), tensor.WithShape(outputSize))

	// input gate weights
	lstm.wi = G.NewMatrix(g, tensor.Float32, G.WithName("Wᵢ"), G.WithShape(hiddenSize, prevSize), G.WithValue(wiT))
	lstm.ui = G.NewMatrix(g, tensor.Float32, G.WithName("Uᵢ"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(uiT))
	lstm.biasI = G.NewVector(g, tensor.Float32, G.WithName("Bᵢ"), G.WithShape(hiddenSize), G.WithValue(biasIT))
	p.Set(`Wᵢ`, lstm.wi)
	p.Set(`Uᵢ`, lstm.ui)
	p.Set(`Bᵢ`, lstm.biasI)

	// output gate weights
	lstm.wo = G.NewMatrix(g, tensor.Float32, G.WithName("Wₒ"), G.WithShape(hiddenSize, prevSize), G.WithValue(woT))
	lstm.uo = G.NewMatrix(g, tensor.Float32, G.WithName("Uₒ"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(uoT))
	lstm.biasO = G.NewVector(g, tensor.Float32, G.WithName("Bₒ"), G.WithShape(hiddenSize), G.WithValue(biasOT))
	p.Set(`Wₒ`, lstm.wo)
	p.Set(`Uₒ`, lstm.uo)
	p.Set(`Bₒ`, lstm.biasO)

	// forget gate weights
	lstm.wf = G.NewMatrix(g, tensor.Float32, G.WithName("Wf"), G.WithShape(hiddenSize, prevSize), G.WithValue(wfT))
	lstm.uf = G.NewMatrix(g, tensor.Float32, G.WithName("Uf"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(ufT))
	lstm.biasF = G.NewVector(g, tensor.Float32, G.WithName("Bf"), G.WithShape(hiddenSize), G.WithValue(biasFT))
	p.Set(`Wf`, lstm.wf)
	p.Set(`Uf`, lstm.uf)
	p.Set(`Bf`, lstm.biasF)

	// cell write
	lstm.wc = G.NewMatrix(g, tensor.Float32, G.WithName("Wc"), G.WithShape(hiddenSize, prevSize), G.WithValue(wcT))
	lstm.uc = G.NewMatrix(g, tensor.Float32, G.WithName("Uc"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(ucT))
	lstm.biasC = G.NewVector(g, tensor.Float32, G.WithName("bc"), G.WithShape(hiddenSize), G.WithValue(biasCT))
	p.Set(`Wc`, lstm.wc)
	p.Set(`Uc`, lstm.uc)
	p.Set(`Bc`, lstm.biasC)

	// Output vector
	lstm.wy = G.NewMatrix(g, tensor.Float32, G.WithName("Wy"), G.WithShape(outputSize, hiddenSize), G.WithValue(wyT))
	lstm.biasY = G.NewVector(g, tensor.Float32, G.WithName("by"), G.WithShape(outputSize), G.WithValue(biasYT))
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

	// input gate weights
	m.wi = back.Wi
	m.ui = back.Ui
	m.biasI = back.BiasI

	// output gate weights
	m.wo = back.Wo
	m.uo = back.Uo
	m.biasO = back.BiasO

	// forget gate weights
	m.wf = back.Wf
	m.uf = back.Uf
	m.biasF = back.BiasF

	// cell write
	m.wc = back.Wc
	m.uc = back.Uc
	m.biasC = back.BiasC

	// Output vector
	m.wy = back.Wy
	m.biasY = back.BiasY
	return m
}

// NewModel creates a new model
func NewModel(inputSize, outputSize int, hiddenSize int) *Model {
	return newModelFromBackends(initBackends(inputSize, outputSize, hiddenSize))
}
