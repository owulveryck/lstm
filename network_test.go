package lstm

import (
	"io/ioutil"
	"math"
	"os"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/encoding/dot"
	"gorgonia.org/tensor"
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

func TestCost(t *testing.T) {
	dim := 3
	g := gorgonia.NewGraph()
	n := &Network{
		Y: []*gorgonia.Node{
			gorgonia.NewVector(g, float, gorgonia.WithName("yy0"),
				gorgonia.WithShape(dim)),
			gorgonia.NewVector(g, float, gorgonia.WithName("yy1"),
				gorgonia.WithShape(dim)),
		},
	}
	y := []*gorgonia.Node{
		gorgonia.NewVector(g, float, gorgonia.WithName("y0"),
			gorgonia.WithShape(dim)),
		gorgonia.NewVector(g, float, gorgonia.WithName("y1"),
			gorgonia.WithShape(dim)),
	}
	yT1 := tensor.NewDense(float, []int{dim}, tensor.WithBacking([]float64{1, 0, 0}))
	yT2 := tensor.NewDense(float, []int{dim}, tensor.WithBacking([]float64{0, 1, 0}))
	yyT1 := tensor.NewDense(float, []int{dim}, tensor.WithBacking([]float64{0.5, 0.1, 0.4}))
	yyT2 := tensor.NewDense(float, []int{dim}, tensor.WithBacking([]float64{0.2, 0.6, 0.2}))
	gorgonia.Let(n.Y[0], yyT1)
	gorgonia.Let(n.Y[1], yyT2)
	gorgonia.Let(y[0], yT1)
	gorgonia.Let(y[1], yT2)
	cost, err := n.Cost(y)
	if err != nil {
		t.Fatal(err)
	}
	vm := gorgonia.NewTapeMachine(g)
	err = vm.RunAll()
	if err != nil {
		t.Fatal(err)
	}
	v, ok := cost.Value().Data().(float64)
	if !ok {
		t.Fatal("Expected a scalar value for cost")
	}
	if v != -(math.Log(0.5) + math.Log(0.6)) {
		t.Fatal(v)
	}

}
func TestCost_err(t *testing.T) {
	dim := 3
	g := gorgonia.NewGraph()
	n := &Network{
		Y: []*gorgonia.Node{
			gorgonia.NewVector(g, float, gorgonia.WithName("yy0"),
				gorgonia.WithShape(dim)),
			gorgonia.NewVector(g, float, gorgonia.WithName("yy1"),
				gorgonia.WithShape(dim)),
		},
	}
	y := []*gorgonia.Node{
		gorgonia.NewVector(g, float, gorgonia.WithName("y1"),
			gorgonia.WithShape(dim)),
	}
	_, err := n.Cost(y)
	if err == nil {
		t.Fail()
	}
}
