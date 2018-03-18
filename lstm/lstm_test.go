package lstm

import (
	"testing"

	G "gorgonia.org/gorgonia"
)

func TestForwardStep(t *testing.T) {
	model := newModelFromBackends(testBackends(5, 5, 100))
	tset := &testSet{
		values: [][]float32{
			[]float32{1, 0, 0, 0, 0},
		}}
	_, _, err := model.forwardStep(tset, model.prevHidden, model.prevCell, 0)
	if err != nil {
		t.Fatal(err)
	}
	//fmt.Println(model.g.ToDot())
	machine := G.NewLispMachine(model.g, G.ExecuteFwdOnly(), G.WithLogger(nil))
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	/*
		for _, computedVector := range tset.GetComputedVectors() {
			t.Log(computedVector.Value().Data().([]float32))
		}
	*/
}
