package text

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func initLearnables(nodes []*gorgonia.Node) {
	for i := 0; i < len(nodes); i++ {
		currentNode := nodes[i]
		t := tensor.NewDense(gorgonia.Float64,
			currentNode.Shape(),
			tensor.WithBacking(gorgonia.Gaussian(0, 0.08)(gorgonia.Float64, currentNode.Shape()...)),
		)
		gorgonia.Let(currentNode, t)
	}
}
