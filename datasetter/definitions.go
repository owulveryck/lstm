package datasetter

import G "gorgonia.org/gorgonia"

// ReadWriter is an interface that can Read and returns a oneOfK encoded vector
type ReadWriter interface {
	ReadInputVector(*G.ExprGraph) (*G.Node, error)
	WriteComputedVector(*G.Node) error
	GetComputedVectors() G.Nodes // Should return all the nodes in the correct order
}

// Trainer is a particular dataset that can be used to train a rnn
// it holds expected values
type Trainer interface {
	ReadWriter
	// get the index of the expected output for offset
	// for example is the expected output is []int{0,0,1,0,0}, it returns 2
	GetExpectedValue(offset int) (int, error)
}
