package lstm

import (
	"context"
	"fmt"
	"log"

	G "gorgonia.org/gorgonia"
)

// Predict the next elements that comes after args
// Predict applies the decision function to the array
func (m *Model) Predict(ctx context.Context, args []int, decision func([]float32) int) (<-chan int, error) {
	for _, v := range args {
		if v > m.inputSize {
			return nil, fmt.Errorf("value %v in not in the range of the input neurons")
		}
	}
	feed := make(chan int)
	go func() {
		var prev *lstmOut
		var err error
		for i := 0; true; i++ {
			var id int
			// Make the LSTM memorize the args
			if i < len(args) {
				id = args[i]
			}

			if prev, err = m.forwardPass(id, prev); err != nil {
				panic(err)
			}
			g := m.g.SubgraphRoots(prev.probs)
			machine := G.NewLispMachine(g, G.ExecuteFwdOnly())
			machine.ForceCPU()
			if err := machine.RunAll(); err != nil {
				log.Printf("ERROR1 while predicting with %p %+v", machine, err)
			}
			if i < len(args) {
				continue
			}

			select {
			case <-ctx.Done():
				m.g.UnbindAllNonInputs()
				close(feed)
				return // returning not to leak the goroutine
			case feed <- decision(prev.probs.Value().Data().([]float32)):
			}
			m.g.UnbindAllNonInputs()
		}
	}()
	return feed, nil
}
