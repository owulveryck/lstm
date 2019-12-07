package main

import (
	"bytes"
	"testing"

	"github.com/owulveryck/lstm"
)

func TestRun(t *testing.T) {
	config := configuration{
		HiddenSize: 10,
		Epoch:      4,
		BatchSize:  5,
		Step:       1,
		Learnrate:  1e-3,
		L2reg:      1e-5,
		ClipVal:    5.0,
	}
	sample := bytes.NewReader([]byte(`abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘abc⌘`))
	dict := getVocabulary(sample)
	vectorSize := len(dict)

	nn := lstm.NewLSTM(vectorSize, config.HiddenSize)
	nn.Dict = dict
	initLearnables(nn.Learnables())
	err := run(nn, sample, config)
	if err != nil {
		t.Fatal(err)
	}
}
