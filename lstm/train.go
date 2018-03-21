package lstm

import (
	"context"
	"io"

	"github.com/owulveryck/charRNN/datasetter"
	G "gorgonia.org/gorgonia"
)

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

// Train ...
func (m *Model) Train(ctx context.Context, dset datasetter.FullTrainer, solver G.Solver) error {
	for {
		trainer, err := dset.GetTrainer()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		cost, _, err := m.cost(trainer)
		if err != nil {
			return err
		}
		g := m.g.SubgraphRoots(cost)
		machine := G.NewLispMachine(g)
		if err := machine.RunAll(); err != nil {
			return err
		}
		solver.Step(G.Nodes{
			m.biasC,
			m.biasF,
			m.biasI,
			m.biasO,
			m.biasY,
			m.uc,
			m.uf,
			m.ui,
			m.uo,
			m.wc,
			m.wf,
			m.wi,
			m.wo,
			m.wy})
	}

	return nil
}
