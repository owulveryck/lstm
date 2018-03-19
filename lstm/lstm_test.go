package lstm

import (
	"testing"

	G "gorgonia.org/gorgonia"
)

func TestForwardStep(t *testing.T) {
	model := newModelFromBackends(testBackends(5, 4, 100))
	tset := &testSet{
		values: [][]float32{
			[]float32{1, 0, 0, 0, 0},
			[]float32{0, 1, 0, 0, 0},
			[]float32{0, 0, 1, 0, 0},
			[]float32{0, 0, 0, 1, 0},
			[]float32{0, 0, 0, 0, 1},
		}}
	_, _, err := model.forwardStep(tset, model.prevHidden, model.prevCell, 0)
	if err != nil {
		t.Fatal(err)
	}
	machine := G.NewLispMachine(model.g, G.ExecuteFwdOnly())
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	for _, computedVector := range tset.GetComputedVectors() {
		t.Log(computedVector.Value().Data().([]float32))
	}
}

func TestCost(t *testing.T) {
	model := newModelFromBackends(testBackends(5, 5, 100))
	tset := &testSet{
		values: [][]float32{
			[]float32{1, 0, 0, 0, 0},
			[]float32{0, 1, 0, 0, 0},
			[]float32{0, 0, 1, 0, 0},
			[]float32{0, 0, 0, 1, 0},
			[]float32{0, 0, 0, 0, 1},
		},
		expectedValues: []int{1, 2, 3, 4, 0},
	}
	_, _, err := model.cost(tset)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		machine := G.NewLispMachine(model.g, G.ExecuteFwdOnly())
		if err := machine.RunAll(); err != nil {
			t.Fatal(err)
		}
		for _, computedVector := range tset.GetComputedVectors() {
			t.Log(computedVector.Value().Data().([]float32))
		}
	}

}
