package parser

import (
	G "gorgonia.org/gorgonia"
)

// Parser is a structure that parses an expression and returns the corresponding gorgonia node
type Parser struct {
	dico map[string]*G.Node
	g    *G.ExprGraph
}

// NewParser ...
func NewParser(g *G.ExprGraph) *Parser {
	return &Parser{
		dico: make(map[string]*G.Node),
		g:    g,
	}
}

func (p *Parser) Get(ident string) *G.Node {
	return p.dico[ident]
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
	l.g = p.g
	gorgoniaParse(l)
	if l.err != nil {
		return nil, l.err
	}
	return l.result, nil
}
