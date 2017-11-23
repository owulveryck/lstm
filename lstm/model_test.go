package lstm

import (
	"fmt"
	"testing"
)

func TestMarshalUnmarshal(t *testing.T) {
	model := NewModel(5, 5, []int{100, 100, 100})
	b, err := model.MarshalBinary()
	if err != nil {
		t.Fatal("Cannot marshal", err)
	}
	modelRestored := new(Model)
	err = modelRestored.UnmarshalBinary(b)
	if err != nil {
		t.Fatal("Cannot Unmarshal", err)
	}
	err = areEquals(model, modelRestored)
	if err != nil {
		t.Fatal(err)
	}
}

func areEquals(a, b *Model) error {
	if len(a.ls) != len(b.ls) {
		return fmt.Errorf("Not the same number of layers expected %v, got %v", len(a.ls), len(b.ls))
	}
	return nil
}
