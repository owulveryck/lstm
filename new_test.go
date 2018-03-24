package lstm

import (
	"fmt"
	"testing"
)

func TestMarshalUnmarshal(t *testing.T) {
	model := NewModel(5, 5, 100)
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

	for i := range a.wi.Value().Data().([]float32) {
		if a.wi.Value().Data().([]float32)[i] != b.wi.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.ui.Value().Data().([]float32) {
		if a.ui.Value().Data().([]float32)[i] != b.ui.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasI.Value().Data().([]float32) {
		if a.biasI.Value().Data().([]float32)[i] != b.biasI.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.wf.Value().Data().([]float32) {
		if a.wf.Value().Data().([]float32)[i] != b.wf.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.uf.Value().Data().([]float32) {
		if a.uf.Value().Data().([]float32)[i] != b.uf.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasF.Value().Data().([]float32) {
		if a.biasF.Value().Data().([]float32)[i] != b.biasF.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.wo.Value().Data().([]float32) {
		if a.wo.Value().Data().([]float32)[i] != b.wo.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.uo.Value().Data().([]float32) {
		if a.uo.Value().Data().([]float32)[i] != b.uo.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasO.Value().Data().([]float32) {
		if a.biasO.Value().Data().([]float32)[i] != b.biasO.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.wc.Value().Data().([]float32) {
		if a.wc.Value().Data().([]float32)[i] != b.wc.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.uc.Value().Data().([]float32) {
		if a.uc.Value().Data().([]float32)[i] != b.uc.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasC.Value().Data().([]float32) {
		if a.biasC.Value().Data().([]float32)[i] != b.biasC.Value().Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	if a.inputSize != b.inputSize {
		return fmt.Errorf("Error")
	}
	if a.outputSize != b.outputSize {
		return fmt.Errorf("Error")
	}
	if a.hiddenSize != b.hiddenSize {
		return fmt.Errorf("Error")
	}
	return nil
}
