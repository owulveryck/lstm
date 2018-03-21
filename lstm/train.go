package lstm

import (
	"context"

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

// TrainingInfos returns info about the current training process
type TrainingInfos struct {
	Step       int
	Perplexity float32
	Cost       float32
	Err        error
	End        bool
}

// Train ...
func (m *Model) Train(ctx context.Context, dset datasetter.FullTrainer, solver G.Solver) chan TrainingInfos {
	infoChan := make(chan TrainingInfos, 0)
	go func() {
		for step := 0; true; step++ {
			trainer, err := dset.GetTrainer()
			if err != nil {
				infoChan <- TrainingInfos{
					Step: step,
					Err:  err,
					End:  true,
				}
				return
			}
			cost, perplexity, err := m.cost(trainer)
			if err != nil {
				infoChan <- TrainingInfos{
					Step: step,
					Err:  err,
					End:  true,
				}
				return
			}
			g := m.g.SubgraphRoots(cost, perplexity)
			machine := G.NewLispMachine(g)
			if err := machine.RunAll(); err != nil {
				infoChan <- TrainingInfos{
					Step: step,
					Err:  err,
					End:  true,
				}
				return
			}
			// send infos about this execution step in a non blocking channel
			select {
			case infoChan <- TrainingInfos{
				Perplexity: perplexity.Value().Data().(float32),
				Cost:       cost.Value().Data().(float32),
				Step:       step,
				Err:        nil,
				End:        false,
			}:
			}
			solver.Step(G.Nodes{
				m.biasC, m.biasF, m.biasI, m.biasO, m.biasY,
				m.uc, m.uf, m.ui, m.uo,
				m.wc, m.wf, m.wi, m.wo, m.wy})
		}
	}()
	return infoChan
}
