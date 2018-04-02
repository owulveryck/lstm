// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

%{

package parser

import (
	"bytes"
	"log"
	"math/big"
        "fmt"
	"unicode/utf8"
        G "gorgonia.org/gorgonia"
)

%}

// gorgoniaSymType ...
%union {
	node *G.Node
}

%type	<node>	expr expr1 expr2 expr3 function

%token '+' '·' '-' '*' '/' '(' ')' '=' 'σ' tanh softmax

%token	<node>	NODE

%%

top:
	expr
	{
                gorgonialex.(*exprLex).result = $1
                
	}

expr:
	expr1
|	'+' expr
	{
		$$ = $2
	}
|	'-' expr
	{
		$$ = G.Must(G.Neg($2))
	}

expr1:
	expr2
|	expr1 '+' expr2
	{
                $$ = G.Must(G.Add($1,$3))
	}
|	expr1 '-' expr2
	{
                $$ = G.Must(G.Sub($1,$3))
	}

expr2:
	expr3
|	expr2 '·' expr3
	{
                $$ = G.Must(G.Mul($1,$3))
	}
|	expr2 '*' expr3
	{
                $$ = G.Must(G.HadamardProd($1,$3))
	}
|	expr2 '/' expr3
	{
                $$ = G.Must(G.Div($1,$3))
	}

expr3:
	NODE
|	'(' expr ')'
	{
		$$ = $2
	}
|       function

function:
     'σ' expr3
      {
                $$ = G.Must(G.Sigmoid($2))
      }
|     tanh expr3
      {
                $$ = G.Must(G.Tanh($2))
      }
|     softmax expr3
      {
                $$ = G.Must(G.SoftMax($2))
      }

%%

// The parser expects the lexer to return 0 on EOF.  Give it a name
// for clarity.
const eof = 0

// The parser uses the type <prefix>Lex as a lexer. It must provide
// the methods Lex(*<prefix>SymType) int and Error(string).
type exprLex struct {
	line []byte
	peek rune
        dico map[string]*G.Node
	g      *G.ExprGraph
        result *G.Node
        err error
}

// Let assigns insert a node into de dictionary represented by the identifier
func (x *exprLex) Let(ident  string, value *G.Node) {
        x.dico[ident] = value
}

// The parser calls this method to get each new token. This
// implementation returns operators and NODE.
func (x *exprLex) Lex(yylval *gorgoniaSymType) int {
	for {
		c := x.next()
		switch c {
		case eof:
			return eof
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			return x.num(c, yylval)
		case 'σ', '+', '-', '*', '·', '/', '(', ')', '=':
			return int(c)

		// Recognize Unicode multiplication and division
		// symbols, returning what the parser expects.
		case '×':
			return '*'
		case '÷':
			return '/'
		case ' ', '\t', '\n', '\r':
		default:
			return x.ident(c, yylval)
		}
	}
}

func isChar(ch rune) bool {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || (ch >= '₀' && ch <= 'ₜ') || ch == 'ᵢ'
}

// Lex a number.
func (x *exprLex) ident(c rune, yylval *gorgoniaSymType) int {
	add := func(b *bytes.Buffer, c rune) {
		if _, err := b.WriteRune(c); err != nil {
			log.Fatalf("WriteRune: %s", err)
		}
	}
	var b bytes.Buffer
	add(&b, c)
	L: for {
		c = x.next()
		switch  {
		case isChar(c):
			add(&b, c)
		default:
			break L
		}
	}
	if c != eof {
		x.peek = c
	}
        // if the token is tanh, return it
        switch b.String() {
        case "tanh":
              return tanh
        case "softmax":
              return softmax
        default:

          // OWK Here we analyse the dictionnary
          yylval.node = &G.Node{}
          val, ok := x.dico[b.String()]
          if !ok {
                  x.Error("Value does not exist in the dictionnary: " + b.String())
                  return eof
          }
          yylval.node = val
        }
        return NODE
}

// Lex a number.
func (x *exprLex) num(c rune, yylval *gorgoniaSymType) int {
	add := func(b *bytes.Buffer, c rune) {
		if _, err := b.WriteRune(c); err != nil {
			log.Fatalf("WriteRune: %s", err)
		}
	}
	var b bytes.Buffer
	add(&b, c)
	L: for {
		c = x.next()
		switch c {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'e', 'E':
			add(&b, c)
		default:
			break L
		}
	}
	if c != eof {
		x.peek = c
	}
	num := &big.Rat{}
	_, ok := num.SetString(b.String())
	if !ok {
		x.Error("Cannot read number: " + b.String())
		return eof
	}

	res, _ := num.Float32()
	yylval.node = G.NewScalar(x.g, G.Float32, G.WithValue(res))

	return NODE
}

// Return the next rune for the lexer.
func (x *exprLex) next() rune {
	if x.peek != eof {
		r := x.peek
		x.peek = eof
		return r
	}
	if len(x.line) == 0 {
		return eof
	}
	c, size := utf8.DecodeRune(x.line)
	x.line = x.line[size:]
	if c == utf8.RuneError && size == 1 {
		log.Print("invalid utf8")
		return x.next()
	}
	return c
}

// The parser calls this method on a parse error.
func (x *exprLex) Error(s string) {
        if x.err != nil {
		x.err = fmt.Errorf("%v\nparse error: %s", x.err, s)
	} else {
		x.err = fmt.Errorf("parse error: %s", s)
	}
}

