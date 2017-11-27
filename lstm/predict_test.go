package lstm

import (
	"context"
	"testing"
)

func TestPredict(t *testing.T) {
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
	_, err := model.Predict(context.TODO(), []int{8, 2, 1}, best)
	if err == nil {
		t.Fail()
	}
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
