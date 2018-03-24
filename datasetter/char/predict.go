package char

import (
	G "gorgonia.org/gorgonia"
)

// Prediction is the based type that can be used as a training dataset
type Prediction struct{}

// ReadInputVector ...
func (p *Prediction) ReadInputVector(g *G.ExprGraph) (*G.Node, error) {
	return nil, nil
}

// WriteComputedVector ...
func (p *Prediction) WriteComputedVector(n *G.Node) error {
	return nil
}

// GetComputedVectors ...
func (p *Prediction) GetComputedVectors() G.Nodes {
	return nil
}
