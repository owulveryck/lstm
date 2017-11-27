package lstm

import (
	"context"
	"log"
	"testing"
)

func TestPredict(t *testing.T) {
	model := NewModel(5, 5, []int{100, 100, 100})
	_, err := model.Predict(context.TODO(), []int{8, 2, 1})
	if err == nil {
		t.Fail()
	}
	f, err := model.Predict(context.TODO(), []int{4, 2, 1})
	if err != nil {
		t.Fatal(err)
	}
	i := 0
	for v := range f {
		i++
		if i > 5 {
			break
		}
		log.Println(v)
	}

}
