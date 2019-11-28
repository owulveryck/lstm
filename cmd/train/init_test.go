package main

import (
	"testing"

	"gorgonia.org/gorgonia"
)

func TestInitLearnables(t *testing.T) {
	g := gorgonia.NewGraph()
	a := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithName("a"), gorgonia.WithShape(2, 2))
	b := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithName("b"), gorgonia.WithShape(2, 2))
	initLearnables([]*gorgonia.Node{a, b})
	_, ok := a.Value().Data().([]float64)
	if !ok {
		t.Fail()
	}
	_, ok = b.Value().Data().([]float64)
	if !ok {
		t.Fail()
	}
}
