package lstm

import (
	"io"
	"log"
	"strconv"
	"strings"

	"github.com/owulveryck/charRNN/datasetter"
	G "gorgonia.org/gorgonia"
)

// forwardStep as described here https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
// It returns the last hidden node and the last cell node
func (m *Model) forwardStep(dataSet datasetter.ReadWriter, prevHidden, prevCell *G.Node, step int) (*G.Node, *G.Node, error) {
	// Read the current input vector
	inputVector, err := dataSet.ReadInputVector(m.g)

	switch {
	case err != nil && err != io.EOF:
		return prevHidden, prevCell, err
	case err == io.EOF:
		return prevHidden, prevCell, nil
	}
	// Helper function for clarity
	script := strings.NewReplacer(`1`, `₁`, `2`, `₂`, `3`, `₃`, `4`, `₄`, `5`, `₅`, `6`, `₆`, `7`, `₇`, `8`, `₈`, `9`, `₉`, `0`, `₀`, `-`, `₋`)
	r := strings.NewReplacer(`ₜ₋₁`, script.Replace(strconv.Itoa(step-1)), `ₜ`, script.Replace(strconv.Itoa(step)))
	set := func(ident, equation string) *G.Node {
		log.Printf("%v=%v", r.Replace(ident), r.Replace(equation))
		res, _ := m.parser.Parse(r.Replace(equation))
		m.parser.Set(r.Replace(ident), res)
		return res
	}

	m.parser.Set(r.Replace(`xₜ`), inputVector)
	if step == 0 {
		m.parser.Set(r.Replace(`hₜ₋₁`), prevHidden)
		m.parser.Set(r.Replace(`cₜ₋₁`), prevCell)
	}
	set(`iₜ`, `σ(Wᵢ·xₜ+Uᵢ·hₜ₋₁+Bᵢ)`)
	set(`fₜ`, `σ(Wf·xₜ+Uf·hₜ₋₁+Bf)`) // dot product made with ctrl+k . M
	set(`oₜ`, `σ(Wₒ·xₜ+Uₒ·hₜ₋₁+Bₒ)`)
	// ċₜis a vector of new candidates value
	set(`ĉₜ`, `tanh(Wc·xₜ+Uc·hₜ₋₁+Bc)`) // c made with ctrl+k c >
	ct := set(`cₜ`, `fₜ*cₜ₋₁+iₜ*ĉₜ`)
	ht := set(`hₜ`, `oₜ*tanh(cₜ)`)
	y, _ := m.parser.Parse(r.Replace(`Wy·hₜ+By`))
	// Apply the softmax function to the output vector
	prob := G.Must(G.SoftMax(y))

	dataSet.WriteComputedVector(prob)
	return m.forwardStep(dataSet, ht, ct, step+1)
}

// the cost function
func (m *Model) cost(dataSet datasetter.Trainer) (cost, perplexity *G.Node, err error) {
	hidden, cell, err := m.forwardStep(dataSet, m.prevHidden, m.prevCell, 0)
	if err != nil {
		return nil, nil, err
	}
	var loss, perp *G.Node
	// Evaluate the cost
	for i, computedVector := range dataSet.GetComputedVectors() {
		expectedValue, err := dataSet.GetExpectedValue(i)
		if err != nil {
			return nil, nil, err
		}
		logprob := G.Must(G.Neg(G.Must(G.Log(computedVector))))
		loss = G.Must(G.Slice(logprob, G.S(expectedValue)))
		log2prob := G.Must(G.Neg(G.Must(G.Log2(computedVector))))
		perp = G.Must(G.Slice(log2prob, G.S(expectedValue)))

		if cost == nil {
			cost = loss
		} else {
			cost = G.Must(G.Add(cost, loss))
		}
		G.WithName("Cost")(cost)

		if perplexity == nil {
			perplexity = perp
		} else {
			perplexity = G.Must(G.Add(perplexity, perp))
		}
	}
	m.prevHidden = hidden
	m.prevCell = cell
	return
}
