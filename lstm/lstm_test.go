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
	model := newModelFromBackends(testBackends(5, 5, 10))
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
	learnrate := 0.01
	l2reg := 1e-6
	clipVal := float64(5)
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))
	for i := 0; i < 100; i++ {
		cost, perplexity, err := model.cost(tset)
		if err != nil {
			t.Fatal(err)
		}
		g := model.g.SubgraphRoots(cost, perplexity)
		machine := G.NewLispMachine(g)
		if err := machine.RunAll(); err != nil {
			t.Fatal(err)
		}
		solver.Step(G.Nodes{
			model.biasC,
			model.biasF,
			model.biasI,
			model.biasO,
			model.biasY,
			model.uc,
			model.uf,
			model.ui,
			model.uo,
			model.wc,
			model.wf,
			model.wi,
			model.wo,
			model.wy})
	}
	getMax := func(a []float32) int {
		max := float32(0)
		idx := 0
		for i, val := range a {
			if val > max {
				idx = i
				max = val
			}
		}
		return idx
	}
	for i, computedVector := range tset.GetComputedVectors() {
		val := getMax(computedVector.Value().Data().([]float32))
		if tset.expectedValues[i] != val {
			t.Log(computedVector.Value().Data().([]float32))
			t.Fatal("Bad result")
		}

	}
}
