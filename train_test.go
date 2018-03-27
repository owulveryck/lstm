package lstm

import (
	"context"
	"io"
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestCost(t *testing.T) {
	model := newModelFromBackends(testBackends(5, 5, 10))
	tset := &testSet{
		values: [][]float32{
			{1, 0, 0, 0, 0},
			{0, 1, 0, 0, 0},
			{0, 0, 1, 0, 0},
			{0, 0, 0, 1, 0},
			{0, 0, 0, 0, 1},
		},
		expectedValues: []int{1, 2, 3, 4, 0},
	}
	learnrate := 0.01
	l2reg := 1e-6
	clipVal := float64(5)
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))
	var hiddenT, cellT tensor.Tensor
	for i := 0; i < 100; i++ {
		if hiddenT == nil {
			hiddenT = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(10))
		}
		if cellT == nil {
			cellT = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(10))
		}
		l := model.newLSTM(hiddenT, cellT)
		cost, perplexity, err := l.cost(tset)
		if err != nil {
			t.Fatal(err)
		}
		g := l.g.SubgraphRoots(cost, perplexity)
		machine := G.NewLispMachine(g)
		if err := machine.RunAll(); err != nil {
			t.Fatal(err)
		}
		solver.Step(G.Nodes{
			l.biasC,
			l.biasF,
			l.biasI,
			l.biasO,
			l.biasY,
			l.uc,
			l.uf,
			l.ui,
			l.uo,
			l.wc,
			l.wf,
			l.wi,
			l.wo,
			l.wy})
		hiddenT = (*l).prevHidden.Value().(tensor.Tensor)
		cellT = (*l).prevCell.Value().(tensor.Tensor)
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

func TestTrain(t *testing.T) {
	model := newModelFromBackends(testBackends(5, 5, 10))
	tset := &testSet{
		values: [][]float32{
			{1, 0, 0, 0, 0},
			{0, 1, 0, 0, 0},
			{0, 0, 1, 0, 0},
			{0, 0, 0, 1, 0},
			{0, 0, 0, 0, 1},
		},
		expectedValues: []int{1, 2, 3, 4, 0},
		maxEpoch:       10,
	}
	learnrate := 0.01
	l2reg := 1e-6
	clipVal := float64(5)
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))

	pause := make(chan struct{})
	infoChan, errc := model.Train(context.TODO(), tset, solver, pause)
	for infos := range infoChan {
		t.Log(infos)
		for _, computedVector := range tset.GetComputedVectors() {
			t.Log(computedVector.Value().Data().([]float32))
		}
	}
	err := <-errc
	if err == io.EOF {
		close(pause)
		return
	}
	if err != nil && err != io.EOF {
		t.Fatal(err)
	}

}
