package lstm

import "gorgonia.org/gorgonia"

func mul(a, b *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Mul(a, b))
}
func add(a, b *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Add(a, b))
}
func sigmoid(a *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Sigmoid(a))
}
func tanh(a *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Tanh(a))
}
func hadamardProd(a, b *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.HadamardProd(a, b))
}
