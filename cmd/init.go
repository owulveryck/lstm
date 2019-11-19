package main

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

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
