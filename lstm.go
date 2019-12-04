package lstm

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var float = tensor.Float64

type LSTM struct {
	VectorSize, HiddenSize int
	G                      *gorgonia.ExprGraph
	Wi, Wf, Wc, Wo         *gorgonia.Node
	Ui, Uf, Uc, Uo         *gorgonia.Node
	Bi, Bf, Bc, Bo         *gorgonia.Node
	Wy, By                 *gorgonia.Node
	Dict                   []rune
}

// NewLSTM returns an empty LSTM. Only the nodes are created, but the graph does not hold any equation.
// It is the responsibility of the user to call NewCell to populate the graph
func NewLSTM(vectorSize, hiddenSize int) *LSTM {
	g := gorgonia.NewGraph()
	wf := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wf"),
		gorgonia.WithShape(hiddenSize, vectorSize))
	wi := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wᵢ"),
		gorgonia.WithShape(hiddenSize, vectorSize))
	wo := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wₒ"),
		gorgonia.WithShape(hiddenSize, vectorSize))
	wc := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wc"),
		gorgonia.WithShape(hiddenSize, vectorSize))
	wy := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wy"),
		gorgonia.WithShape(vectorSize, hiddenSize))
	uf := gorgonia.NewMatrix(g, float, gorgonia.WithName("Uf"),
		gorgonia.WithShape(hiddenSize, hiddenSize))
	ui := gorgonia.NewMatrix(g, float, gorgonia.WithName("Uᵢ"),
		gorgonia.WithShape(hiddenSize, hiddenSize))
	uo := gorgonia.NewMatrix(g, float, gorgonia.WithName("Uₒ"),
		gorgonia.WithShape(hiddenSize, hiddenSize))
	uc := gorgonia.NewMatrix(g, float, gorgonia.WithName("Uc"),
		gorgonia.WithShape(hiddenSize, hiddenSize))
	bf := gorgonia.NewVector(g, float, gorgonia.WithName("bf"),
		gorgonia.WithShape(hiddenSize))
	bi := gorgonia.NewVector(g, float, gorgonia.WithName("bᵢ"),
		gorgonia.WithShape(hiddenSize))
	bo := gorgonia.NewVector(g, float, gorgonia.WithName("bₒ"),
		gorgonia.WithShape(hiddenSize))
	bc := gorgonia.NewVector(g, float, gorgonia.WithName("bc"),
		gorgonia.WithShape(hiddenSize))
	by := gorgonia.NewVector(g, float, gorgonia.WithName("by"),
		gorgonia.WithShape(vectorSize))

	return &LSTM{
		G:  g,
		Wi: wi, Wf: wf, Wo: wo, Wc: wc,
		Ui: ui, Uf: uf, Uo: uo, Uc: uc,
		Bi: bi, Bf: bf, Bo: bo, Bc: bc,
		Wy: wy,
		By: by,

		VectorSize: vectorSize,
		HiddenSize: hiddenSize,
	}
}

func (l *LSTM) learnableNodes() []*gorgonia.Node {
	return []*gorgonia.Node{
		l.Wi, l.Wf, l.Wc, l.Wo,
		l.Ui, l.Uf, l.Uc, l.Uo,
		l.Bi, l.Bf, l.Bc, l.Bo,
		l.Wy, l.By,
	}
}

// newCell to the LSTM network. It takes as input x_t, h_{t-1}, c_{t-1} and returns
// h_t and c_t
func (l *LSTM) NewCell(x, hPrev, cPrev *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node) {
	it := sigmoid(add(add(mul(l.Wi, x), mul(l.Ui, hPrev)), l.Bi))
	ft := sigmoid(add(add(mul(l.Wf, x), mul(l.Uf, hPrev)), l.Bf))
	ot := sigmoid(add(add(mul(l.Wo, x), mul(l.Uo, hPrev)), l.Bo))
	cct := tanh(add(add(mul(l.Wc, x), mul(l.Uc, hPrev)), l.Bc))
	c := add(hadamardProd(ft, cPrev), hadamardProd(it, cct))
	h := hadamardProd(ot, tanh(c))
	return h, c
}

func (l *LSTM) Dense(h *gorgonia.Node) *gorgonia.Node {
	return softmax(add(mul(l.Wy, h), l.By))
}

func (l *LSTM) Learnables() []*gorgonia.Node {
	return []*gorgonia.Node{
		l.Wi, l.Wf, l.Wc, l.Wo,
		l.Ui, l.Uf, l.Uc, l.Uo,
		l.Bi, l.Bf, l.Bc, l.Bo,
		l.Wy, l.By,
	}
}
