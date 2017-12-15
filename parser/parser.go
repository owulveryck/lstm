package parser

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// Parser is a structure that parses an expression and returns the corresponding gorgonia node
type Parser struct {
	dico map[string]*G.Node
}

// NewParser ...
func NewParser() *Parser {
	return &Parser{
		dico: make(map[string]*G.Node),
	}
}

// Set a value to the ident
func (p *Parser) Set(ident string, value *G.Node) {
	p.dico[ident] = value
}

// Parse a string and returns the node
func (p *Parser) Parse(s string) (*G.Node, error) {
	l := &exprLex{}
	l.line = []byte(s)
	l.dico = p.dico
	output := gorgoniaParse(l)
	if output != 0 {
		return nil, fmt.Errorf("error code %v", output)
	}
	return l.result, nil
}
