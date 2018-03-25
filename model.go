package lstm

import (
	"github.com/owulveryck/lstm/parser"
	G "gorgonia.org/gorgonia"
)

// Model holds the structure of the LSTM model
type Model struct {
	g           *G.ExprGraph
	definitions string
	equations   string
	parser      *parser.Parser

	inputSize  int
	outputSize int
	hiddenSize int
	//inputVector *G.Node
	prevHidden *G.Node
	prevCell   *G.Node
}

// NewModel creates a new model from a string.
// the strings represent a forward pass as well as the definitions
// of the weights expressed in unicode.
// The hidden vector must be called `h` and the memory cell `c`
// The input vector and output vector must be called `xₜ` and `yₜ`
// ex:
//  definitions := `
//      iₜ=(Wᵢ·xₜ+Uᵢ·hₜ₋₁+Bᵢ)
//      fₜ=σ(Wf·xₜ+Uf·hₜ₋₁+Bf)
//      oₜ=σ(Wₒ·xₜ+Uₒ·hₜ₋₁+Bₒ)
//      ĉₜ=tanh(Wc·xₜ+Uc·hₜ₋₁+Bc)
//      cₜ=fₜ*cₜ₋₁+iₜ*ĉₜ
//      hₜ=oₜ*tanh(cₜ)
//      y=Wy·hₜ+By
//      xₜ∈R⁶⁵
//      fₜ∈R¹⁰⁰
//      iₜ∈R¹⁰⁰
//      oₜ∈R¹⁰⁰
//      hₜ∈R¹⁰⁰
//      cₜ∈R¹⁰⁰
//      Wᵢ∈R¹⁰⁰x⁶⁵
//      Uᵢ∈R¹⁰⁰x¹⁰⁰
//      Bᵢ∈R¹⁰⁰
//      Wₒ∈R¹⁰⁰x⁶⁵
//      Uₒ∈R¹⁰⁰x¹⁰⁰
//      Bₒ∈R¹⁰⁰
//      Wf∈R¹⁰⁰x⁶⁵
//      Uf∈R¹⁰⁰x¹⁰⁰
//      Bf∈R¹⁰⁰
//      Wc∈R¹⁰⁰x⁶⁵
//      Uc∈R¹⁰⁰x¹⁰⁰
//      Bc∈R¹⁰⁰
//  `
// The subscript 'ₜ' will be replaces at runtime by a number corresponding
// to the step
func NewModel(definitions string) (*Model, error) {
	return nil, nil
}
