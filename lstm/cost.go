package lstm

import (
	"errors"

	"gorgonia.org/gorgonia"
)

// cost evaluates the cost and perplexity between the sources and the expected result
func (m *Model) cost(source, expected []int) (cost, perplexity *gorgonia.Node, n int, err error) {
	var prev *lstmOut
	if len(source) != len(expected) {
		return nil, nil, 0, errors.New("Source and expected shoud have the same length")
	}

	for i := 0; i < len(source); i++ {

		var loss, perp *gorgonia.Node

		if prev, err = m.forwardPass(source[i], prev); err != nil {
			return
		}

		logprob := gorgonia.Must(gorgonia.Neg(gorgonia.Must(gorgonia.Log(prev.probs))))
		loss = gorgonia.Must(gorgonia.Slice(logprob, gorgonia.S(expected[i])))
		log2prob := gorgonia.Must(gorgonia.Neg(gorgonia.Must(gorgonia.Log2(prev.probs))))
		perp = gorgonia.Must(gorgonia.Slice(log2prob, gorgonia.S(expected[i])))

		if cost == nil {
			cost = loss
		} else {
			cost = gorgonia.Must(gorgonia.Add(cost, loss))
		}
		gorgonia.WithName("Cost")(cost)

		if perplexity == nil {
			perplexity = perp
		} else {
			perplexity = gorgonia.Must(gorgonia.Add(perplexity, perp))
		}
	}
	return
}
