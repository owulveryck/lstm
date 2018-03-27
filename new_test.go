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

	for i := range a.wiT.Data().([]float32) {
		if a.wiT.Data().([]float32)[i] != b.wiT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.uiT.Data().([]float32) {
		if a.uiT.Data().([]float32)[i] != b.uiT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasIT.Data().([]float32) {
		if a.biasIT.Data().([]float32)[i] != b.biasIT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.wfT.Data().([]float32) {
		if a.wfT.Data().([]float32)[i] != b.wfT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.ufT.Data().([]float32) {
		if a.ufT.Data().([]float32)[i] != b.ufT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasFT.Data().([]float32) {
		if a.biasFT.Data().([]float32)[i] != b.biasFT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.woT.Data().([]float32) {
		if a.woT.Data().([]float32)[i] != b.woT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.uoT.Data().([]float32) {
		if a.uoT.Data().([]float32)[i] != b.uoT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasOT.Data().([]float32) {
		if a.biasOT.Data().([]float32)[i] != b.biasOT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.wcT.Data().([]float32) {
		if a.wcT.Data().([]float32)[i] != b.wcT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.ucT.Data().([]float32) {
		if a.ucT.Data().([]float32)[i] != b.ucT.Data().([]float32)[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasCT.Data().([]float32) {
		if a.biasCT.Data().([]float32)[i] != b.biasCT.Data().([]float32)[i] {
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
