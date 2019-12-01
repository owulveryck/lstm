package lstm

import (
	"io/ioutil"
	"os"
	"testing"

	"gorgonia.org/gorgonia/encoding/dot"
)

func TestNewNetwork(t *testing.T) {
	vectorSize := 3
	hiddenSize := 2
	nn := NewLSTM(vectorSize, hiddenSize)
	network := NewNetwork(nn, 4)
	if len(network.X) != 4 {
		t.Fail()
	}
}

func TestNetwork_dot(t *testing.T) {
	if _, ok := os.LookupEnv("DOT"); !ok {
		t.SkipNow()
	}
	vectorSize := 3
	hiddenSize := 2
	nn := NewLSTM(vectorSize, hiddenSize)
	NewNetwork(nn, 1)
	b, err := dot.Marshal(nn.G)
	if err != nil {
		t.Fatal(err)
	}
	dotFile, err := ioutil.TempFile("", "network.*.dot")
	if err != nil {
		t.Fatal(err)
	}

	defer os.Remove(dotFile.Name())
	if _, err := dotFile.Write(b); err != nil {
		t.Fatal(err)
	}
	if err := dotFile.Close(); err != nil {
		t.Fatal(err)
	}

	t.Log("content dumped into ", dotFile.Name())
}
