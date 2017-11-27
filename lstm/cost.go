package lstm

import (
	"errors"

	G "gorgonia.org/gorgonia"
)

// cost evaluation between the source and the target result
func (m *Model) cost(source, target []int) (cost, perplexity *G.Node, n int, err error) {
	var prev *lstmOut
	if len(source) != len(target) {
		return nil, nil, 0, errors.New("Source and target shoud have the same length")
	}

	for i := 0; i < len(source); i++ {

		var loss, perp *G.Node

		if prev, err = m.forwardPass(source[i], prev); err != nil {
			return
		}

		logprob := G.Must(G.Neg(G.Must(G.Log(prev.probs))))
		loss = G.Must(G.Slice(logprob, G.S(target[i])))
		log2prob := G.Must(G.Neg(G.Must(G.Log2(prev.probs))))
		perp = G.Must(G.Slice(log2prob, G.S(target[i])))

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
	return
}
