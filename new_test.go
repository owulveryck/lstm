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

	for i := range a.wi {
		if a.wi[i] != b.wi[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.ui {
		if a.ui[i] != b.ui[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasI {
		if a.biasI[i] != b.biasI[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.wf {
		if a.wf[i] != b.wf[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.uf {
		if a.uf[i] != b.uf[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasF {
		if a.biasF[i] != b.biasF[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.wo {
		if a.wo[i] != b.wo[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.uo {
		if a.uo[i] != b.uo[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasO {
		if a.biasO[i] != b.biasO[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.wc {
		if a.wc[i] != b.wc[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.uc {
		if a.uc[i] != b.uc[i] {
			return fmt.Errorf("Error")
		}
	}
	for i := range a.biasC {
		if a.biasC[i] != b.biasC[i] {
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
