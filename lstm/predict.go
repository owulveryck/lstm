package lstm

import (
	"context"
	"log"

	G "gorgonia.org/gorgonia"
)

// Predict the next elements that comes after args
func (m *Model) Predict(ctx context.Context, args []int) <-chan int {
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

			//sampledID := sample(prev.probs.Value())
			select {
			case <-ctx.Done():
				m.g.UnbindAllNonInputs()
				return // returning not to leak the goroutine
			case feed <- 0:
			}
			m.g.UnbindAllNonInputs()
		}
	}()
	return feed
}
