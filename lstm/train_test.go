package lstm

import (
	"context"
	"testing"

	"gorgonia.org/gorgonia"
)

func TestTrain(t *testing.T) {
	best := func(vals []float32) int {
		best := float32(0)
		idx := 0
		for i, v := range vals {
			if v > best {
				idx = i

			}
		}
		return idx
	}
	model := NewModel(5, 5, []int{100, 100, 100})

	var l2reg = 0.000001
	var learnrate = 0.01
	var clipVal = 5.0
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(learnrate), gorgonia.WithL2Reg(l2reg), gorgonia.WithClip(clipVal))
	model.Train([]int{4, 2, 1}, []int{2, 1, 3}, solver)
	ctx, cancel := context.WithCancel(context.Background())
	f, err := model.Predict(ctx, []int{4, 2, 1}, best)
	if err != nil {
		t.Fatal(err)
	}
	i := 0
	for v := range f {
		i++
		if i > 5 {
			cancel()
		}
		t.Log(v)
	}

}
