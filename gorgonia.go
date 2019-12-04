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
func softmax(a *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.SoftMax(a))
}
func logarithm(a *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Log(a))
}
func log2(a *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Log2(a))
}
func neg(a *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Neg(a))
}
func mean(a *gorgonia.Node) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Mean(a))
}
