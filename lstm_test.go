package lstm

import (
	"testing"

	G "gorgonia.org/gorgonia"
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
	lstm := model.newLSTM()
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
