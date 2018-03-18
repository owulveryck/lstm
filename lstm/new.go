package lstm

import (
	"bytes"
	"encoding/gob"

	"github.com/owulveryck/charRNN/parser"
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

	wc     *G.Node
	uc     *G.Node
	biasC  *G.Node
	parser *parser.Parser

	inputSize  int
	outputSize int
	hiddenSize int
	//inputVector *G.Node
	prevHidden *G.Node
	prevCell   *G.Node
}

// backends holds the informations to be saved
type backends struct {
	InputSize  int
	OutputSize int
	HiddenSize int
	Wi         []float32
	Ui         []float32
	BiasI      []float32

	Wf    []float32
	Uf    []float32
	BiasF []float32

	Wo    []float32
	Uo    []float32
	BiasO []float32

	Wc    []float32
	Uc    []float32
	BiasC []float32
}

// MarshalBinary for backup. This function saves the content of the weights matrices and the biais but not the graph structure
func (m Model) MarshalBinary() ([]byte, error) {
	var bkp backends
	bkp.InputSize = m.inputSize
	bkp.OutputSize = m.outputSize
	bkp.HiddenSize = m.hiddenSize
	bkp.Wi = m.wi.Value().Data().([]float32)
	bkp.Ui = m.ui.Value().Data().([]float32)
	bkp.BiasI = m.biasI.Value().Data().([]float32)
	bkp.Wf = m.wf.Value().Data().([]float32)
	bkp.Uf = m.uf.Value().Data().([]float32)
	bkp.BiasF = m.biasF.Value().Data().([]float32)
	bkp.Wo = m.wo.Value().Data().([]float32)
	bkp.Uo = m.uo.Value().Data().([]float32)
	bkp.BiasO = m.biasO.Value().Data().([]float32)
	bkp.Wc = m.wc.Value().Data().([]float32)
	bkp.Uc = m.uc.Value().Data().([]float32)
	bkp.BiasC = m.biasC.Value().Data().([]float32)
	var output bytes.Buffer
	enc := gob.NewEncoder(&output)
	err := enc.Encode(bkp)
	return output.Bytes(), err
}

// UnmarshalBinary for restore
func (m *Model) UnmarshalBinary(data []byte) error {
	bkp := new(backends)
	output := bytes.NewBuffer(data)
	dec := gob.NewDecoder(output)
	err := dec.Decode(bkp)
	if err != nil {
		return err
	}
	*m = *newModelFromBackends(bkp)
	return nil
}

// initBackends returns weights initialisation
func initBackends(inputSize, outputSize int, hiddenSize int) *backends {
	var back backends
	back.InputSize = inputSize
	back.OutputSize = outputSize
	back.HiddenSize = hiddenSize
	back.Wi = G.Gaussian32(0.0, 0.08, hiddenSize, inputSize)
	back.Ui = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
	back.BiasI = make([]float32, hiddenSize)
	back.Wo = G.Gaussian32(0.0, 0.08, hiddenSize, inputSize)
	back.Uo = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
	back.BiasO = make([]float32, hiddenSize)
	back.Wf = G.Gaussian32(0.0, 0.08, hiddenSize, inputSize)
	back.Uf = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
	back.BiasF = make([]float32, hiddenSize)
	back.Wc = G.Gaussian32(0.0, 0.08, hiddenSize, inputSize)
	back.Uc = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
	back.BiasC = make([]float32, hiddenSize)
	return &back
}

func newModelFromBackends(back *backends) *Model {
	m := new(Model)
	g := G.NewGraph()
	m.hiddenSize = back.HiddenSize
	m.inputSize = back.InputSize
	m.outputSize = back.OutputSize

	prevSize := back.InputSize
	hiddenSize := back.HiddenSize
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
