package lstm

import (
	"context"

	"github.com/owulveryck/lstm/datasetter"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Predict ...
func (m *Model) Predict(ctx context.Context, dataSet datasetter.ReadWriter) error {
	lstm := m.newLSTM()
	// We need an empty memory to start...
	hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(m.hiddenSize))
	cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(m.hiddenSize))
	prevHidden := G.NewVector(lstm.g, tensor.Float32, G.WithName("hₜ₋₁"), G.WithShape(m.hiddenSize), G.WithValue(hiddenT))
	prevCell := G.NewVector(lstm.g, tensor.Float32, G.WithName("Cₜ₋₁"), G.WithShape(m.hiddenSize), G.WithValue(cellT))
	_, _, err := lstm.forwardStep(dataSet, prevHidden, prevCell, 0)
	if err != nil {
		return err
	}
	g := lstm.g.SubgraphRoots(dataSet.GetComputedVectors()...)
	machine := G.NewLispMachine(g, G.ExecuteFwdOnly())
	err = machine.RunAll()
	return err

}
