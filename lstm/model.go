package lstm

import (
	"bytes"
	"encoding/gob"
	"strconv"

	"github.com/chewxy/gorgonia/tensor"
	G "gorgonia.org/gorgonia"
)

type layer struct {
	wix   *G.Node
	wih   *G.Node
	biasI *G.Node

	wfx   *G.Node
	wfh   *G.Node
	biasF *G.Node

	wox   *G.Node
	woh   *G.Node
	biasO *G.Node

	wcx   *G.Node
	wch   *G.Node
	biasC *G.Node
}

// Model holds the structure of the LSTM model
type Model struct {
	g  *G.ExprGraph
	ls []*layer

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
	Wix         [][]float32
	Wih         [][]float32
	BiasI       [][]float32

	Wfx   [][]float32
	Wfh   [][]float32
	BiasF [][]float32

	Wox   [][]float32
	Woh   [][]float32
	BiasO [][]float32

	Wcx   [][]float32
	Wch   [][]float32
	BiasC [][]float32

	Whd   []float32
	BiasD []float32
}

// MarshalBinary for backup. This function saves the content of the weights matrices and the biais but not the graph structure
func (m Model) MarshalBinary() ([]byte, error) {
	numLayers := len(m.ls)
	var bkp backends
	bkp.InputSize = m.inputSize
	bkp.OutputSize = m.outputSize
	bkp.HiddenSizes = m.hiddenSizes
	bkp.Wix = make([][]float32, numLayers)
	bkp.Wih = make([][]float32, numLayers)
	bkp.BiasI = make([][]float32, numLayers)
	bkp.Wfx = make([][]float32, numLayers)
	bkp.Wfh = make([][]float32, numLayers)
	bkp.BiasF = make([][]float32, numLayers)
	bkp.Wox = make([][]float32, numLayers)
	bkp.Woh = make([][]float32, numLayers)
	bkp.BiasO = make([][]float32, numLayers)
	bkp.Wcx = make([][]float32, numLayers)
	bkp.Wch = make([][]float32, numLayers)
	bkp.BiasC = make([][]float32, numLayers)
	for i := 0; i < numLayers; i++ {
		bkp.Wix[i] = m.ls[i].wix.Value().Data().([]float32)
		bkp.Wih[i] = m.ls[i].wih.Value().Data().([]float32)
		bkp.BiasI[i] = m.ls[i].biasI.Value().Data().([]float32)
		bkp.Wfx[i] = m.ls[i].wfx.Value().Data().([]float32)
		bkp.Wfh[i] = m.ls[i].wfh.Value().Data().([]float32)
		bkp.BiasF[i] = m.ls[i].biasF.Value().Data().([]float32)
		bkp.Wox[i] = m.ls[i].wox.Value().Data().([]float32)
		bkp.Woh[i] = m.ls[i].woh.Value().Data().([]float32)
		bkp.BiasO[i] = m.ls[i].biasO.Value().Data().([]float32)
		bkp.Wcx[i] = m.ls[i].wcx.Value().Data().([]float32)
		bkp.Wch[i] = m.ls[i].wch.Value().Data().([]float32)
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
	numLayers := len(hiddenSizes)
	back.Wix = make([][]float32, numLayers)
	back.Wih = make([][]float32, numLayers)
	back.BiasI = make([][]float32, numLayers)
	back.Wfx = make([][]float32, numLayers)
	back.Wfh = make([][]float32, numLayers)
	back.BiasF = make([][]float32, numLayers)
	back.Wox = make([][]float32, numLayers)
	back.Woh = make([][]float32, numLayers)
	back.BiasO = make([][]float32, numLayers)
	back.Wcx = make([][]float32, numLayers)
	back.Wch = make([][]float32, numLayers)
	back.BiasC = make([][]float32, numLayers)
	for depth := 0; depth < numLayers; depth++ {
		prevSize := inputSize
		if depth > 0 {
			prevSize = hiddenSizes[depth-1]
		}
		hiddenSize := hiddenSizes[depth]
		back.Wix[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, prevSize)
		back.Wih[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
		back.BiasI[depth] = make([]float32, hiddenSize)
		back.Wox[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, prevSize)
		back.Woh[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
		back.BiasO[depth] = make([]float32, hiddenSize)
		back.Wfx[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, prevSize)
		back.Wfh[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
		back.BiasF[depth] = make([]float32, hiddenSize)
		back.Wcx[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, prevSize)
		back.Wch[depth] = G.Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)
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
		l := new(layer)
		m.ls = append(m.ls, l) // add layer to model

		layerID := strconv.Itoa(depth)

		// input gate weights
		wixT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wix[depth]))
		wihT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Wih[depth]))
		//biasIT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		biasIT := tensor.New(tensor.WithBacking(back.BiasI[depth]), tensor.WithShape(hiddenSize))

		l.wix = G.NewMatrix(g, tensor.Float32, G.WithName("wix_"+layerID), G.WithShape(hiddenSize, prevSize), G.WithValue(wixT))
		l.wih = G.NewMatrix(g, tensor.Float32, G.WithName("wih_"+layerID), G.WithShape(hiddenSize, hiddenSize), G.WithValue(wihT))
		l.biasI = G.NewVector(g, tensor.Float32, G.WithName("biasI_"+layerID), G.WithShape(hiddenSize), G.WithValue(biasIT))

		// output gate weights

		woxT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wox[depth]))
		wohT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Woh[depth]))
		biasOT := tensor.New(tensor.WithBacking(back.BiasO[depth]), tensor.WithShape(hiddenSize))

		l.wox = G.NewMatrix(g, tensor.Float32, G.WithName("wox_"+layerID), G.WithShape(hiddenSize, prevSize), G.WithValue(woxT))
		l.woh = G.NewMatrix(g, tensor.Float32, G.WithName("woh_"+layerID), G.WithShape(hiddenSize, hiddenSize), G.WithValue(wohT))
		l.biasO = G.NewVector(g, tensor.Float32, G.WithName("biasO_"+layerID), G.WithShape(hiddenSize), G.WithValue(biasOT))

		// forget gate weights

		wfxT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wfx[depth]))
		wfhT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Wfh[depth]))
		biasFT := tensor.New(tensor.WithBacking(back.BiasF[depth]), tensor.WithShape(hiddenSize))

		l.wfx = G.NewMatrix(g, tensor.Float32, G.WithName("wfx_"+layerID), G.WithShape(hiddenSize, prevSize), G.WithValue(wfxT))
		l.wfh = G.NewMatrix(g, tensor.Float32, G.WithName("wfh_"+layerID), G.WithShape(hiddenSize, hiddenSize), G.WithValue(wfhT))
		l.biasF = G.NewVector(g, tensor.Float32, G.WithName("biasF_"+layerID), G.WithShape(hiddenSize), G.WithValue(biasFT))

		// cell write

		wcxT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(back.Wcx[depth]))
		wchT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(back.Wch[depth]))
		biasCT := tensor.New(tensor.WithBacking(back.BiasC[depth]), tensor.WithShape(hiddenSize))

		l.wcx = G.NewMatrix(g, tensor.Float32, G.WithName("wcx_"+layerID), G.WithShape(hiddenSize, prevSize), G.WithValue(wcxT))
		l.wch = G.NewMatrix(g, tensor.Float32, G.WithName("wch_"+layerID), G.WithShape(hiddenSize, hiddenSize), G.WithValue(wchT))
		l.biasC = G.NewVector(g, tensor.Float32, G.WithName("biasC_"+layerID), G.WithShape(hiddenSize), G.WithValue(biasCT))

		// this is to simulate a default "previous" state
		hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		hidden := G.NewVector(g, tensor.Float32, G.WithName("prevHidden_"+layerID), G.WithShape(hiddenSize), G.WithValue(hiddenT))
		cell := G.NewVector(g, tensor.Float32, G.WithName("prevCell_"+layerID), G.WithShape(hiddenSize), G.WithValue(cellT))

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
	m.inputVector = G.NewVector(g, tensor.Float32, G.WithName("inputVector_"), G.WithShape(back.InputSize), G.WithValue(dummyInputVec))
	m.prevHiddens = hiddens
	m.prevCells = cells

	m.g = g
	return m
}

// NewModel creates a new model
func NewModel(inputSize, outputSize int, hiddenSizes []int) *Model {
	return newModelFromBackends(initBackends(inputSize, outputSize, hiddenSizes))
}
