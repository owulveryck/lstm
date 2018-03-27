package lstm

import (
	"bytes"
	"encoding/gob"

	G "gorgonia.org/gorgonia"
)

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

	Wy    []float32
	BiasY []float32
}

// MarshalBinary for backup. This function saves the content of the weights matrices and the biais but not the graph structure
func (m Model) MarshalBinary() ([]byte, error) {
	var bkp backends
	bkp.InputSize = m.inputSize
	bkp.OutputSize = m.outputSize
	bkp.HiddenSize = m.hiddenSize
	bkp.Wi = m.wiT.Data().([]float32)
	bkp.Ui = m.uiT.Data().([]float32)
	bkp.BiasI = m.biasIT.Data().([]float32)
	bkp.Wf = m.wfT.Data().([]float32)
	bkp.Uf = m.ufT.Data().([]float32)
	bkp.BiasF = m.biasFT.Data().([]float32)
	bkp.Wo = m.woT.Data().([]float32)
	bkp.Uo = m.uoT.Data().([]float32)
	bkp.BiasO = m.biasOT.Data().([]float32)
	bkp.Wc = m.wcT.Data().([]float32)
	bkp.Uc = m.ucT.Data().([]float32)
	bkp.BiasC = m.biasCT.Data().([]float32)
	bkp.Wy = m.wyT.Data().([]float32)
	bkp.BiasY = m.biasYT.Data().([]float32)
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
	back.Wy = G.Gaussian32(0.0, 0.08, outputSize, hiddenSize)
	back.BiasY = make([]float32, outputSize)
	return &back
}
