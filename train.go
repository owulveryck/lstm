package lstm

import (
	"context"
	"errors"
	"sync"

	"github.com/owulveryck/lstm/datasetter"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
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
}

// Train the model
func (m *Model) Train(ctx context.Context, dset datasetter.FullTrainer, solver G.Solver, pauseChan <-chan struct{}) (<-chan TrainingInfos, <-chan error) {
	infoChan := make(chan TrainingInfos, 0)
	step := 0
	errc := make(chan error, 1)
	var wg sync.WaitGroup
	wg.Add(1)
	paused := false

	go func() {
		if len(pauseChan) != 0 {
			errc <- errors.New("pauseChan must not be buffered")
			wg.Done()
			return
		}
		for {
			select {
			case <-ctx.Done():
				errc <- nil
				wg.Done()
				return
			case <-pauseChan:
				paused = true
			default:
				if paused {
					<-pauseChan
					paused = false
				}
				step++
				trainer, err := dset.GetTrainer()
				if err != nil {
					errc <- err
					wg.Done()
					return
				}
				cost, perplexity, err := m.cost(trainer)
				if err != nil {
					errc <- err
					wg.Done()
					return
				}
				g := m.g.SubgraphRoots(cost, perplexity)
				//g := m.g
				machine := G.NewTapeMachine(g, G.BindDualValues(m.biasC, m.biasF, m.biasI, m.biasO, m.biasY,
					m.uc, m.uf, m.ui, m.uo,
					m.wc, m.wf, m.wi, m.wo, m.wy), G.TraceExec())
				if err := machine.RunAll(); err != nil {
					errc <- err
					wg.Done()
					return
				}
				// send infos about this execution step in a non blocking channel
				select {
				case infoChan <- TrainingInfos{
					Perplexity: perplexity.Value().Data().(float32),
					Cost:       cost.Value().Data().(float32),
					Step:       step,
				}:
				}
				solver.Step(G.Nodes{
					m.biasC, m.biasF, m.biasI, m.biasO, m.biasY,
					m.uc, m.uf, m.ui, m.uo,
					m.wc, m.wf, m.wi, m.wo, m.wy})
				// Before unbinding, save the value of the last hidden state and the last cell
				// and the recreate them. They will be wiped out because they are linked in the graph and
				// therefore not considerer as being "inputs"
				hiddenValue := make([]float32, m.hiddenSize)
				cellValue := make([]float32, m.hiddenSize)
				copy(hiddenValue, m.prevHidden.Value().Data().([]float32))
				copy(cellValue, m.prevCell.Value().Data().([]float32))
				m.g = g
				m.g.UnbindAllNonInputs()
				hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(m.hiddenSize), tensor.WithBacking(hiddenValue))
				cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(m.hiddenSize), tensor.WithBacking(cellValue))
				m.prevHidden = G.NewVector(g, tensor.Float32, G.WithName("hₜ₋₁"), G.WithShape(m.hiddenSize), G.WithValue(hiddenT))
				m.prevCell = G.NewVector(g, tensor.Float32, G.WithName("Cₜ₋₁"), G.WithShape(m.hiddenSize), G.WithValue(cellT))
			}
		}
	}()
	go func() {
		wg.Wait()
		close(infoChan)
	}()
	return infoChan, errc
}
