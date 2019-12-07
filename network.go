package lstm

import (
	"errors"
	"fmt"

	"gorgonia.org/gorgonia"
)

// Network is a top-level structure that holds a complete multilayer LSTM
type Network struct {
	lstm *LSTM
	X    []*gorgonia.Node
	H    []*gorgonia.Node
	C    []*gorgonia.Node
	Y    []*gorgonia.Node
}

func NewNetwork(nn *LSTM, layers int) *Network {
	vectorSize := nn.VectorSize
	hiddenSize := nn.HiddenSize
	x := make([]*gorgonia.Node, layers)
	y := make([]*gorgonia.Node, layers)
	h := make([]*gorgonia.Node, layers+1)
	c := make([]*gorgonia.Node, layers+1)
	h[0] = gorgonia.NewVector(nn.G, gorgonia.Float64, gorgonia.WithName("hₜ"),
		gorgonia.WithShape(hiddenSize))
	c[0] = gorgonia.NewVector(nn.G, gorgonia.Float64, gorgonia.WithName("cₜ"),
		gorgonia.WithShape(hiddenSize))
	for i := 0; i < layers; i++ {
		x[i] = gorgonia.NewVector(nn.G, gorgonia.Float64, gorgonia.WithName(fmt.Sprintf("xₜ+%v", i)),
			gorgonia.WithShape(vectorSize))
		h[i+1], c[i+1] = nn.NewCell(x[i], h[i], c[i])
		y[i] = nn.Dense(h[i+1])
	}
	return &Network{
		lstm: nn,
		X:    x,
		H:    h,
		C:    c,
		Y:    y,
	}
}

func (n *Network) CrossEntropy(target []*gorgonia.Node) (cost *gorgonia.Node, err error) {
	if len(target) != len(n.Y) {
		return nil, errors.New("target and Y are of different size")
	}
	var loss *gorgonia.Node
	for i := 0; i < len(n.Y); i++ {
		loss = neg(mean(hadamardProd(target[i], logarithm(n.Y[i]))))

		if cost == nil {
			cost = loss
		} else {
			cost = add(cost, loss)
		}
		gorgonia.WithName("Cost")(cost)

	}
	return cost, nil
}
