package lstm

import (
	"math"
	"runtime"

	G "gorgonia.org/gorgonia"
)

func (m *Model) inputs() (retVal G.Nodes) {
	for _, l := range m.ls {
		lin := G.Nodes{
			l.wix,
			l.wih,
			l.biasI,
			l.wfx,
			l.wfh,
			l.biasF,
			l.wox,
			l.woh,
			l.biasO,
			l.wcx,
			l.wch,
			l.biasC,
		}

		retVal = append(retVal, lin...)
	}

	retVal = append(retVal, m.whd)
	retVal = append(retVal, m.biasD)
	return
}

// Train the neural network by giving him a source and an expected target
// solver is the algo used to adapt the gradient
func (m *Model) Train(source, target []int, evaluate bool, solver G.Solver) (retCost, retPerp float32, err error) {
	defer runtime.GC()

	var cost *G.Node
	var perp *G.Node
	var n int

	cost, perp, n, err = m.cost(source, target)
	cost, _, n, err = m.cost(source, target)
	if err != nil {
		return
	}

	var readCost *G.Node
	var readPerp *G.Node
	var costVal G.Value
	var perpVal G.Value

	var g *G.ExprGraph
	if evaluate {
		readPerp = G.Read(perp, &perpVal)
		readCost = G.Read(cost, &costVal)
		g = m.g.SubgraphRoots(cost, readPerp, readCost)
	} else {
		g = m.g.SubgraphRoots(cost)
	}

	machine := G.NewLispMachine(g, G.UseCudaFor("tanh", "mul", "exp", "sigmoid"))
	if err = machine.RunAll(); err != nil {
		return
	}
	machine.UnbindAll()

	err = solver.Step(m.inputs())
	if err != nil {
		return
	}

	if evaluate {
		if sv, ok := perpVal.(G.Scalar); ok {
			v := sv.Data().(float32)
			retPerp = float32(math.Pow(2, float64(v)/(float64(n)-1)))
		}
		if cv, ok := costVal.(G.Scalar); ok {
			retCost = cv.Data().(float32)
		}
	}
	return
}
