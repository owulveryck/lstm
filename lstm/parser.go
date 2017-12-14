package lstm

import (
	"fmt"
	"strings"

	G "gorgonia.org/gorgonia"
)

// Parser represents a parser.
type Parser struct {
	s   *Scanner
	buf struct {
		tok Token  // last read token
		lit string // last read literal
		n   int    // buffer size (max=1)
	}
	// dico holds a correspondance between an IDENT and a node representation
	dico map[string]*G.Node
}

// Let assigns the litteral to a node
func (p *Parser) Let(a string, b *G.Node) {
	p.dico[a] = b
}

// NewParser returns a new instance of Parser.
func NewParser() *Parser {
	return &Parser{
		dico: make(map[string]*G.Node),
	}

}

// scan returns the next token from the underlying scanner.
// If a token has been unscanned then read that instead.
func (p *Parser) scan() (tok Token, lit string) {
	// If we have a token on the buffer, then return it.
	if p.buf.n != 0 {
		p.buf.n = 0
		return p.buf.tok, p.buf.lit
	}

	// Otherwise read the next token from the scanner.
	tok, lit = p.s.Scan()

	// Save it to the buffer in case we unscan later.
	p.buf.tok, p.buf.lit = tok, lit

	return
}

// unscan pushes the previously read token back onto the buffer.
func (p *Parser) unscan() { p.buf.n = 1 }

// scanIgnoreWhitespace scans the next non-whitespace token.
func (p *Parser) scanIgnoreWhitespace() (tok Token, lit string) {
	tok, lit = p.scan()
	if tok == WS {
		tok, lit = p.scan()
	}
	return
}

// Parse ...
func (p *Parser) Parse(input string) (*G.Node, error) {

	p.s = NewScanner(strings.NewReader(input))
	// additionStack will hold all the nodes to be added at the very end
	var additionStack []*G.Node

	// We should read a sequence of IDENT OPERATOR IDENT
	for {
		var iden1, iden2 string
		var tok Token
		// Read a field.
		tok, iden1 = p.scanIgnoreWhitespace()
		if tok != IDENT {
			return nil, fmt.Errorf("1. found %q, expected field", iden1)
		}
		operator, _ := p.scanIgnoreWhitespace()
		if operator != DOTPRODUCT && operator != PRODUCT && operator != ADDITION {
			return nil, fmt.Errorf("2. found %q, expected field", iden1)
		}
		tok, iden2 = p.scanIgnoreWhitespace()
		if tok != IDENT {
			return nil, fmt.Errorf("3. found %q, expected field", iden1)
		}
		switch operator {
		case DOTPRODUCT:
			additionStack = append(additionStack, G.Must(G.Mul(p.dico[iden1], p.dico[iden2])))
		case PRODUCT:
			additionStack = append(additionStack, G.Must(G.HadamardProd(p.dico[iden1], p.dico[iden2])))
		case ADDITION:
			additionStack = append(additionStack, p.dico[iden2])
		}

		// If the next token is not a comma then break the loop.
		if tok, _ = p.scanIgnoreWhitespace(); tok == EOF {
			p.unscan()
			break
		}
		p.unscan()
	}
	var res *G.Node
	for i := 0; i < len(additionStack)-1; i++ {
		if res == nil {
			res = additionStack[i]
		}
		res = G.Must(G.Add(res, additionStack[i+1]))

	}
	// Return the successfully parsed statement.
	return res, nil
}
