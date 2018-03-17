package lstm

import (
	"io"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// dataSetIO is an interface that can Read and returns a oneOfK encoded vector
type dataSetIO interface {
	ReadInputVec() ([]float32, error)
	WriteVec([]float32) error
}

// Forward pass as described here https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
// It returns the last hidden node and the last cell node
func (m *Model) fwd(dataSet dataSetIO, prevHidden, prevCell *G.Node) (*G.Node, *G.Node, error) {
	// Read the current input vector
	inputVectorValue, err := dataSet.ReadInputVec()
	inputTensor := tensor.New(tensor.Of(tensor.Float32), tensor.WithBacking(inputVectorValue))
	inputVector := G.NewVector(m.g, tensor.Float32, G.WithName("xₜ"), G.WithShape(m.inputSize), G.WithValue(inputTensor))

	switch {
	case err != nil && err != io.EOF:
		return prevHidden, prevCell, err
	case err == io.EOF:
		return prevHidden, prevCell, nil
	}
	// Helper function for clarity
	set := func(ident, equation string) *G.Node {
		res, _ := m.parser.Parse(equation)
		m.parser.Set(ident, res)
		return res
	}

	m.parser.Set(`xₜ`, inputVector)
	m.parser.Set(`hₜ₋₁`, prevHidden)
	m.parser.Set(`cₜ₋₁`, prevCell)
	set(`iₜ`, `σ(Wᵢ·xₜ+Uᵢ·hₜ₋₁+Bᵢ)`)
	set(`fₜ`, `σ(Wf·xₜ+Uf·hₜ₋₁+Bf)`) // dot product made with ctrl+k . M
	set(`oₜ`, `σ(Wₒ·xₜ+Uₒ·hₜ₋₁+Bₒ)`)
	// ċₜis a vector of new candidates value
	set(`ĉₜ`, `tanh(Wc·xₜ+Uc·hₜ₋₁+Bc)`) // c made with ctrl+k c >
	ct := set(`cₜ`, `fₜ*cₜ₋₁+iₜ*ĉₜ`)
	set(`hc`, `tanh(cₜ)`)
	ht, _ := m.parser.Parse(`oₜ*hc`)
	// Apply the softmax function to the output vector
	prob := G.Must(G.SoftMax(ht))

	dataSet.WriteVec(prob.Value().Data().([]float32))
	return m.fwd(dataSet, prob, ct)
}
