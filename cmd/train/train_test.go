package main

import (
	"bytes"
	"testing"

	"github.com/owulveryck/lstm"
)

func TestTrain(t *testing.T) {
	config := configuration{
		HiddenSize: 10,
		Epoch:      5,
		BatchSize:  5,
		Step:       1,
	}
	sample := bytes.NewReader([]byte(`abc⌘abc⌘abc⌘abc⌘abc⌘`))
	dict := getVocabulary(sample)
	vectorSize := len(dict)

	nn := lstm.NewLSTM(vectorSize, config.HiddenSize)
	nn.Dict = dict
	initLearnables(nn.Learnables())
	err := train(nn, sample, config)
	if err != nil {
		t.Fatal(err)
	}
}
