package main

import (
	"gorgonia.org/gorgonia"
)

type lstm struct {
	VectorSize, HiddenSize int
	G                      *gorgonia.ExprGraph
	Wi, Wf, Wc, Wo         *gorgonia.Node
	Wd                     *gorgonia.Node
	Ui, Uf, Uc, Uo         *gorgonia.Node
	Bi, Bf, Bc, Bo         *gorgonia.Node
	Dict                   map[rune]int
}

func newLSTM(vectorSize, hiddenSize int) *lstm {
	g := gorgonia.NewGraph()
	// Declarations
	//	xt := gorgonia.NewVector(g, float, gorgonia.WithName("xₜ"),
	//		gorgonia.WithShape(vectorSize))
	//	htprev := gorgonia.NewVector(g, float, gorgonia.WithName("hₜ₋₁"),
	//		gorgonia.WithShape(hiddenSize))
	//ctprev := gorgonia.NewVector(g, float, gorgonia.WithName("cₜ₋₁"),
	//	gorgonia.WithShape(hiddenSize))
	wf := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wf"),
		gorgonia.WithShape(hiddenSize, vectorSize))
	wi := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wᵢ"),
		gorgonia.WithShape(hiddenSize, vectorSize))
	wo := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wₒ"),
		gorgonia.WithShape(hiddenSize, vectorSize))
	wc := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wc"),
		gorgonia.WithShape(hiddenSize, vectorSize))
	wd := gorgonia.NewMatrix(g, float, gorgonia.WithName("Wd"),
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

	/*
		yt := gorgonia.Must(gorgonia.SoftMax(
			gorgonia.Must(
				gorgonia.Mul(
					wd, ht,
				))))
	*/

	return &lstm{
		G:  g,
		Wi: wi, Wf: wf, Wo: wo, Wc: wc,
		Wd: wd,
		Ui: ui, Uf: uf, Uo: uo, Uc: uc,
		Bi: bi, Bf: bf, Bo: bo, Bc: bc,

		VectorSize: vectorSize,
		HiddenSize: hiddenSize,
	}
}

func (l *lstm) learnableNodes() []*gorgonia.Node {
	return []*gorgonia.Node{
		l.Wi, l.Wf, l.Wc, l.Wo,
		l.Wd,
		l.Ui, l.Uf, l.Uc, l.Uo,
		l.Bi, l.Bf, l.Bc, l.Bo,
	}
}

// newCell to the LSTM network
func (l *lstm) newCell(x, hPrev, cPrev *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node) {
	it := gorgonia.Must(
		gorgonia.Sigmoid(
			gorgonia.Must(
				gorgonia.Add(
					gorgonia.Must(
						gorgonia.Add(
							gorgonia.Must(gorgonia.Mul(l.Wi, x)),
							gorgonia.Must(gorgonia.Mul(l.Ui, hPrev)))),
					l.Bi,
				))))
	ft := gorgonia.Must(
		gorgonia.Sigmoid(
			gorgonia.Must(
				gorgonia.Add(
					gorgonia.Must(
						gorgonia.Add(
							gorgonia.Must(gorgonia.Mul(l.Wf, x)),
							gorgonia.Must(gorgonia.Mul(l.Uf, hPrev)))),
					l.Bf,
				))))
	ot := gorgonia.Must(
		gorgonia.Sigmoid(
			gorgonia.Must(
				gorgonia.Add(
					gorgonia.Must(
						gorgonia.Add(
							gorgonia.Must(gorgonia.Mul(l.Wo, x)),
							gorgonia.Must(gorgonia.Mul(l.Uo, hPrev)))),
					l.Bo,
				))))
	cct := gorgonia.Must(
		gorgonia.Tanh(
			gorgonia.Must(
				gorgonia.Add(
					gorgonia.Must(
						gorgonia.Add(
							gorgonia.Must(gorgonia.Mul(l.Wc, x)),
							gorgonia.Must(gorgonia.Mul(l.Uc, hPrev)))),
					l.Bc,
				))))
	c := gorgonia.Must(gorgonia.Add(
		gorgonia.Must(gorgonia.HadamardProd(ft, cPrev)),
		gorgonia.Must(gorgonia.HadamardProd(it, cct)),
	))
	h := gorgonia.Must(
		gorgonia.HadamardProd(
			ot,
			gorgonia.Must(gorgonia.Tanh(c)),
		))
	return h, c
}
