package lstm

import G "gorgonia.org/gorgonia"

func tanh(l *lstm, equation string) *G.Node {
	val, _ := l.parser.Parse(equation)
	return G.Must(G.Tanh(val))

}
func σ(l *lstm, equation string) *G.Node {
	val, _ := l.parser.Parse(equation)
	return G.Must(G.Sigmoid(val))

}
