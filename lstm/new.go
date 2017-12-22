package lstm

import (
	"bytes"
	"encoding/gob"
	"strconv"

	"github.com/owulveryck/charRNN/parser"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type lstm struct {
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
}

// Model holds the structure of the LSTM model
type Model struct {
	g  *G.ExprGraph
	ls []*lstm

	whd   *G.Node
	biasD *G.Node

	inputSize   int
	outputSize  int
	hiddenSizes []int
	inputVector *G.Node
	prevHiddens G.Nodes
	prevCells   G.Nodes
}

// backends holds the informations to be saved
type backends struct {
	InputSize   int
	OutputSize  int
	HiddenSizes []int
	wi          [][]float32
	ui          [][]float32
	BiasI       [][]float32

	wf    [][]float32
	uf    [][]float32
	BiasF [][]float32

	wo    [][]float32
	uo    [][]float32
	BiasO [][]float32

	wc    [][]float32
	uc    [][]float32
	BiasC [][]float32

	Whd   []float32
	BiasD []float32
}

// MarshalBinary for backup. This function saves the content of the weights matrices and the biais but not the graph structure
func (m Model) MarshalBinary() ([]byte, error) {
	numlstms := len(m.ls)
	var bkp backends
	bkp.InputSize = m.inputSize
	bkp.OutputSize = m.outputSize
	bkp.HiddenSizes = m.hiddenSizes
	bkp.wi = make([][]float32, numlstms)
	bkp.ui = make([][]float32, numlstms)
	bkp.BiasI = make([][]float32, numlstms)
	bkp.wf = make([][]float32, numlstms)
	bkp.uf = make([][]float32, numlstms)
	bkp.BiasF = make([][]float32, numlstms)
	bkp.wo = make([][]float32, numlstms)
	bkp.uo = make([][]float32, numlstms)
	bkp.BiasO = make([][]float32, numlstms)
	bkp.wc = make([][]float32, numlstms)
	bkp.uc = make([][]float32, numlstms)
	bkp.BiasC = make([][]float32, numlstms)
	for i := 0; i < numlstms; i++ {
		bkp.wi[i] = m.ls[i].wi.Value().Data().([]float32)
		bkp.ui[i] = m.ls[i].ui.Value().Data().([]float32)
		bkp.BiasI[i] = m.ls[i].biasI.Value().Data().([]float32)
		bkp.wf[i] = m.ls[i].wf.Value().Data().([]float32)
		bkp.uf[i] = m.ls[i].uf.Value().Data().([]float32)
		bkp.BiasF[i] = m.ls[i].biasF.Value().Data().([]float32)
		bkp.wo[i] = m.ls[i].wo.Value().Data().([]float32)
		bkp.uo[i] = m.ls[i].uo.Value().Data().([]float32)
		bkp.BiasO[i] = m.ls[i].biasO.Value().Data().([]float32)
		bkp.wc[i] = m.ls[i].wc.Value().Data().([]float32)
		bkp.uc[i] = m.ls[i].uc.Value().Data().([]float32)
		bkp.BiasC[i] = m.ls[i].biasC.Value().Data().([]float32)
	}
	bkp.Whd = m.whd.Value().Data().([]float32)
	bkp.BiasD = m.biasD.Value().Data().([]float32)
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
func initBackends(inputSize, outputSize int, hiddenSizes []int) *backends {
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
		back.wi[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, prevSize)
		back.ui[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
		back.BiasI[depth] = make([]float32, hiddenSize)
		back.wo[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, prevSize)
		back.uo[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
		back.BiasO[depth] = make([]float32, hiddenSize)
		back.wf[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, prevSize)
		back.uf[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
		back.BiasF[depth] = make([]float32, hiddenSize)
		back.wc[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, prevSize)
		back.uc[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
		back.BiasC[depth] = make([]float32, hiddenSize)
	}
	back.Whd = G.Gaussian32(0.0, 0.08, back.OutputSize, hiddenSizes[len(hiddenSizes)-1])
	back.BiasD = make([]float32, outputSize)
	return &back
}
func newModelFromBackends(back *backends) *Model {
	m := new(Model)
	g := G.NewGraph()
	m.hiddenSizes = back.HiddenSizes
	m.inputSize = back.InputSize
	m.outputSize = back.OutputSize

	var hiddens, cells []*G.Node
	for depth := 0; depth < len(back.HiddenSizes); depth++ {
		prevSize := back.InputSize
		if depth > 0 {
			prevSize = back.HiddenSizes[depth-1]
		}
		hiddenSize := back.HiddenSizes[depth]
		p := parser.NewParser(g)
		l := new(lstm)
		l.parser = p
		m.ls = append(m.ls, l) // add lstm to model

		layerID := strconv.Itoa(depth)

		// input gate weights
		wiT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.wi[depth]))
		uiT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.ui[depth]))
		//biasIT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		biasIT := tensor.New(tensor.WithBacking(back.BiasI[depth]), tensor.WithShape(hiddenSize))

		l.wi = G.NewMatrix(g, tensor.Float32, G.WithName("Wᵢ["+layerID+"]"), G.WithShape(hiddenSize, prevSize), G.WithValue(wiT))
		l.ui = G.NewMatrix(g, tensor.Float32, G.WithName("Uᵢ["+layerID+"]"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(uiT))
		l.biasI = G.NewVector(g, tensor.Float32, G.WithName("Bᵢ["+layerID+"]"), G.WithShape(hiddenSize), G.WithValue(biasIT))
		p.Set(`Wᵢ`, l.wi)
		p.Set(`Uᵢ`, l.ui)
		p.Set(`Bᵢ`, l.biasI)

		// output gate weights

		woT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.wo[depth]))
		uoT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.uo[depth]))
		biasOT := tensor.New(tensor.WithBacking(back.BiasO[depth]), tensor.WithShape(hiddenSize))

		l.wo = G.NewMatrix(g, tensor.Float32, G.WithName("Wₒ["+layerID+"]"), G.WithShape(hiddenSize, prevSize), G.WithValue(woT))
		l.uo = G.NewMatrix(g, tensor.Float32, G.WithName("Uₒ["+layerID+"]"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(uoT))
		l.biasO = G.NewVector(g, tensor.Float32, G.WithName("Bₒ["+layerID+"]"), G.WithShape(hiddenSize), G.WithValue(biasOT))
		p.Set(`Wₒ`, l.wo)
		p.Set(`Uₒ`, l.uo)
		p.Set(`Bₒ`, l.biasO)

		// forget gate weights

		wfT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.wf[depth]))
		ufT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.uf[depth]))
		biasFT := tensor.New(tensor.WithBacking(back.BiasF[depth]), tensor.WithShape(hiddenSize))

		l.wf = G.NewMatrix(g, tensor.Float32, G.WithName("Wf["+layerID+"]"), G.WithShape(hiddenSize, prevSize), G.WithValue(wfT))
		l.uf = G.NewMatrix(g, tensor.Float32, G.WithName("Uf["+layerID+"]"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(ufT))
		l.biasF = G.NewVector(g, tensor.Float32, G.WithName("Bf["+layerID+"]"), G.WithShape(hiddenSize), G.WithValue(biasFT))
		p.Set(`Wf`, l.wf)
		p.Set(`Uf`, l.uf)
		p.Set(`Bf`, l.biasF)

		// cell write

		wcT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.wc[depth]))
		ucT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.uc[depth]))
		biasCT := tensor.New(tensor.WithBacking(back.BiasC[depth]), tensor.WithShape(hiddenSize))

		l.wc = G.NewMatrix(g, tensor.Float32, G.WithName("Wc["+layerID+"]"), G.WithShape(hiddenSize, prevSize), G.WithValue(wcT))
		l.uc = G.NewMatrix(g, tensor.Float32, G.WithName("Uc["+layerID+"]"), G.WithShape(hiddenSize, hiddenSize), G.WithValue(ucT))
		l.biasC = G.NewVector(g, tensor.Float32, G.WithName("bc["+layerID+"]"), G.WithShape(hiddenSize), G.WithValue(biasCT))
		p.Set(`Wc`, l.wc)
		p.Set(`Uc`, l.uc)
		p.Set(`Bc`, l.biasC)

		// this is to simulate a default "previous" state
		hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		hidden := G.NewVector(g, tensor.Float32, G.WithName("hₜ₋₁["+layerID+"]"), G.WithShape(hiddenSize), G.WithValue(hiddenT))
		cell := G.NewVector(g, tensor.Float32, G.WithName("Cₜ₋₁["+layerID+"]"), G.WithShape(hiddenSize), G.WithValue(cellT))

		hiddens = append(hiddens, hidden)
		cells = append(cells, cell)
	}

	lastHiddenSize := back.HiddenSizes[len(back.HiddenSizes)-1]

	whdT := tensor.New(tensor.WithShape(back.OutputSize, lastHiddenSize), tensor.WithBacking(back.Whd))
	biasDT := tensor.New(tensor.WithBacking(back.BiasD), tensor.WithShape(back.OutputSize))

	m.whd = G.NewMatrix(g, tensor.Float32, G.WithName("whd_"), G.WithShape(back.OutputSize, lastHiddenSize), G.WithValue(whdT))
	m.biasD = G.NewVector(g, tensor.Float32, G.WithName("biasD_"), G.WithShape(back.OutputSize), G.WithValue(biasDT))

	// these are to simulate a previous state
	dummyInputVec := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(back.InputSize)) // zeroes
	m.inputVector = G.NewVector(g, tensor.Float32, G.WithName("xₜ"), G.WithShape(back.InputSize), G.WithValue(dummyInputVec))
	m.prevHiddens = hiddens
	m.prevCells = cells

	m.g = g
	return m
}

// NewModel creates a new model
func NewModel(inputSize, outputSize int, hiddenSizes []int) *Model {
	return newModelFromBackends(initBackends(inputSize, outputSize, hiddenSizes))
}
