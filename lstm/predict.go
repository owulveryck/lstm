package lstm

import (
	"context"

	"github.com/owulveryck/charRNN/datasetter"
	G "gorgonia.org/gorgonia"
)

// Predict ...
func (m *Model) Predict(ctx context.Context, dataSet datasetter.ReadWriter) error {
	hidden, cell, err := m.forwardStep(dataSet, m.prevHidden, m.prevCell, 0)
	if err != nil {
		return err
	}
	g := m.g.SubgraphRoots(dataSet.GetComputedVectors()...)
	machine := G.NewLispMachine(g, G.ExecuteFwdOnly())
	if err := machine.RunAll(); err != nil {
		return err
	}

	m.prevHidden = hidden
	m.prevCell = cell
	return nil
}
