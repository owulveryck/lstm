package lstm

import G "gorgonia.org/gorgonia"

func tanh(m *Model, equation string) *G.Node {
	val, _ := m.parser.Parse(equation)
	return G.Must(G.Tanh(val))

}
func σ(m *Model, equation string) *G.Node {
	val, _ := m.parser.Parse(equation)
	return G.Must(G.Sigmoid(val))

}
