package lstm

import (
	"math"
	"runtime"

	"gorgonia.org/gorgonia"
)

func (m *Model) inputs() (retVal gorgonia.Nodes) {
	for _, l := range m.ls {
		lin := gorgonia.Nodes{
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
func (m *Model) Train(source, target []int, solver gorgonia.Solver) (retCost, retPerp float32, err error) {
	defer runtime.GC()

	var cost, perp *gorgonia.Node
	var n int

	cost, perp, n, err = m.cost(source, target)
	if err != nil {
		return
	}

	var readCost *gorgonia.Node
	var readPerp *gorgonia.Node
	var costVal gorgonia.Value
	var perpVal gorgonia.Value

	var g *gorgonia.ExprGraph
	//if iter%100 == 0 {
	readPerp = gorgonia.Read(perp, &perpVal)
	readCost = gorgonia.Read(cost, &costVal)
	g = m.g.SubgraphRoots(cost, readPerp, readCost)
	//} else {
	//	g = m.g.SubgraphRoots(cost)
	//}

	machine := gorgonia.NewLispMachine(g)
	if err = machine.RunAll(); err != nil {
		return
	}
	machine.UnbindAll()

	err = solver.Step(m.inputs())
	if err != nil {
		return
	}

	//if iter%100 == 0 {
	if sv, ok := perpVal.(gorgonia.Scalar); ok {
		v := sv.Data().(float32)
		retPerp = float32(math.Pow(2, float64(v)/(float64(n)-1)))
	}
	if cv, ok := costVal.(gorgonia.Scalar); ok {
		retCost = cv.Data().(float32)
	}
	//}
	return
}
