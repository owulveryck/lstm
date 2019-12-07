package main

import (
	"bytes"
	"testing"

	"github.com/owulveryck/lstm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestRun(t *testing.T) {
	config := configuration{
		HiddenSize: 50,
		Epoch:      50,
		BatchSize:  5,
		Step:       1,
		Learnrate:  1e-1,
		L2reg:      1e-5,
		ClipVal:    5.0,
	}
	ds := []byte(`abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘`)
	//sample := bytes.NewReader([]byte(`abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘`))
	sample := bytes.NewReader(ds)
	dict := getVocabulary(sample)
	vectorSize := len(dict)

	nn := lstm.NewLSTM(vectorSize, config.HiddenSize)
	nn.Dict = dict
	initLearnables(nn.Learnables())
	err := run(nn, sample, config)
	if err != nil {
		t.Fatal(err)
	}
	var backup bytes.Buffer

	err = nn.Save(&backup)
	if err != nil {
		t.Fatal(err)
	}
	bkp := backup.Bytes()
	t.Log(string(ds))
	for i := 0; i < 4; i++ {
		buf := bytes.NewBuffer(bkp)
		predictNN, err := lstm.NewTrainedLSTM(buf)
		model := lstm.NewNetwork(predictNN, 1)
		backend := make([]float64, len(dict))
		backend[i] = 1
		xT := tensor.NewDense(tensor.Float64, []int{model.X[0].Shape()[0], 1}, tensor.WithBacking(backend))
		gorgonia.Let(model.X[0], xT)
		shape := model.H[0].Shape()
		hT := tensor.NewDense(tensor.Float64, shape, tensor.WithBacking(make([]float64, shape[0])))
		gorgonia.Let(model.H[0], hT)
		shape = model.C[0].Shape()
		cT := tensor.NewDense(tensor.Float64, shape, tensor.WithBacking(make([]float64, shape[0])))
		gorgonia.Let(model.C[0], cT)

		vm := gorgonia.NewTapeMachine(predictNN.G)
		err = vm.RunAll()
		if err != nil {
			t.Fatal(err)
		}
		vm.Close()
		err = setValues(model.X, xT)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%v -> %v", string(predictNN.Dict[i]), string(predictNN.Dict[getIdx(model.Y[0].Value().Data().([]float64))]))
	}
}

func getIdx(f []float64) int {
	max := f[0]
	var idx int
	for i := 0; i < len(f); i++ {
		if f[i] > max {
			max = f[i]
			idx = i
		}
	}
	return idx
}
