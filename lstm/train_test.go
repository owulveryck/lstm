package lstm

import (
	"context"
	"os"
	"testing"

	"gorgonia.org/gorgonia"
)

func TestTrain(t *testing.T) {
	if os.Getenv("LSTM_CONVERGE") == "" {
		t.Skip("skipping test; $LSTM_CONVERGE not set")
	}

	best := func(vals []float32) int {
		best := float32(0)
		idx := 0
		for i, v := range vals {
			if v > best {
				best = v
				idx = i

			}
		}
		return idx
	}
	model := NewModel(2, 2, []int{10, 10, 10})

	var l2reg = 0.000001
	var learnrate = 0.01
	var clipVal = 5.0
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(learnrate), gorgonia.WithL2Reg(l2reg), gorgonia.WithClip(clipVal))
	var cost, perp float32
	var err error
	cost = 1
	for i := 0; cost > 1e-5; i++ {
		cost, perp, err = model.Train([]int{1, 1, 0, 1, 1}, []int{1, 0, 1, 1, 0}, solver)
		if err != nil {
			t.Fatal(err)
		}
		if i%200 == 0 {
			t.Logf("[%v] cost: %v, perplexity: %v", i, cost, perp)
		}
	}
	t.Logf("cost: %v, perplexity: %v", cost, perp)

	ctx, cancel := context.WithCancel(context.Background())
	f, err := model.Predict(ctx, []int{1, 1}, best)
	if err != nil {
		t.Fatal(err)
	}
	i := 0
	for v := range f {
		i++
		if i > 6 {
			cancel()
		}
		t.Log(v)
	}

}
