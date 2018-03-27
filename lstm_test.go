package lstm

import (
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestForwardStep(t *testing.T) {
	model := newModelFromBackends(testBackends(5, 4, 100))
	tset := &testSet{
		values: [][]float32{
			{1, 0, 0, 0, 0},
			{0, 1, 0, 0, 0},
			{0, 0, 1, 0, 0},
			{0, 0, 0, 1, 0},
			{0, 0, 0, 0, 1},
		}}
	hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(model.hiddenSize))
	cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(model.hiddenSize))
	lstm := model.newLSTM(hiddenT, cellT)
	//lstm := model.newLSTM()
	_, _, err := lstm.forwardStep(tset, lstm.prevHidden, lstm.prevCell, 0)
	if err != nil {
		t.Fatal(err)
	}
	machine := G.NewLispMachine(lstm.g, G.ExecuteFwdOnly())
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	for _, computedVector := range tset.GetComputedVectors() {
		t.Log(computedVector.Value().Data().([]float32))
	}
}
