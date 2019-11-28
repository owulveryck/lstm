package main

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/owulveryck/lstm"
)

func TestSave(t *testing.T) {
	tmpfile, err := ioutil.TempFile("", "test")
	if err != nil {
		t.Fatal(err)
	}

	defer os.Remove(tmpfile.Name()) // clean up

	nn := lstm.NewLSTM(10, 10)
	initLearnables(nn.Learnables())
	err = save(nn, tmpfile.Name())
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(tmpfile.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	nn2, err := lstm.NewTrainedLSTM(f)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(nn.Learnables()); i++ {
		for j := 0; j < len(nn.Learnables()[i].Value().Data().([]float64)); j++ {
			if nn.Learnables()[i].Value().Data().([]float64)[j] != nn2.Learnables()[i].Value().Data().([]float64)[j] {
				t.Fail()
			}
		}
	}
}
