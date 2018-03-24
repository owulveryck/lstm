package char

import G "gorgonia.org/gorgonia"

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

// TrainingSet ...
type TrainingSet struct{}

// ReadInputVector ...
func (t *TrainingSet) ReadInputVector(g *G.ExprGraph) (*G.Node, error) {
	return nil, nil
}

// WriteComputedVector ...
func (t *TrainingSet) WriteComputedVector(n *G.Node) error {
	return nil
}

// GetComputedVectors ..
func (t *TrainingSet) GetComputedVectors() G.Nodes {
	return nil
}

// GetExpectedValue ...
func (t *TrainingSet) GetExpectedValue(offset int) (int, error) {
	return 0, nil
}

// GetTrainingSet is returning the self object
func (t *TrainingSet) GetTrainingSet() (*TrainingSet, error) {
	return t, nil
}
