package lstm

import (
	"bytes"
	"reflect"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestSaveLoad(t *testing.T) {
	vectorSize := 100
	hiddenSize := 100
	lstm := NewLSTM(vectorSize, hiddenSize)
	var backup bytes.Buffer
	err := lstm.Save(&backup)
	if err == nil {
		t.Fatal(err)
	}
	backup.Reset()
	initLearnables(lstm.learnableNodes(), gorgonia.Gaussian(0, 0.08))
	err = lstm.Save(&backup)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := NewTrainedLSTM(&backup)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(lstm.learnableNodes()); i++ {
		if !reflect.DeepEqual(lstm.learnableNodes()[i].Value().Data(), restored.learnableNodes()[i].Value().Data()) {
			t.Fatalf("%v and %v differ\n", lstm.learnableNodes()[i].Value(), restored.learnableNodes()[i].Value())
		}
	}
}

func initLearnables(nodes []*gorgonia.Node, initFn gorgonia.InitWFn) {
	for i := 0; i < len(nodes); i++ {
		currentNode := nodes[i]
		t := tensor.NewDense(float,
			currentNode.Shape(),
			tensor.WithBacking(initFn(float, currentNode.Shape()...)),
		)
		gorgonia.Let(currentNode, t)
	}
}
