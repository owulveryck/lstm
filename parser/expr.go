//line expr.y:6
package parser

import __yyfmt__ "fmt"

//line expr.y:7
import (
	"bytes"
	"fmt"
	G "gorgonia.org/gorgonia"
	"log"
	"math/big"
	"unicode/utf8"
)

//line expr.y:21
type gorgoniaSymType struct {
	yys  int
	node *G.Node
}

const tanh = 57346
const softmax = 57347
const NODE = 57348

var gorgoniaToknames = [...]string{
	"$end",
	"error",
	"$unk",
	"'+'",
	"'·'",
	"'-'",
	"'*'",
	"'/'",
	"'('",
	"')'",
	"'='",
	"'σ'",
	"tanh",
	"softmax",
	"NODE",
}
var gorgoniaStatenames = [...]string{}

const gorgoniaEofCode = 1
const gorgoniaErrCode = 2
const gorgoniaInitialStackSize = 16

//line expr.y:99

// The parser expects the lexer to return 0 on EOF.  Give it a name
// for clarity.
const eof = 0

// The parser uses the type <prefix>Lex as a lexer. It must provide
// the methods Lex(*<prefix>SymType) int and Error(string).
type exprLex struct {
	line   []byte
	peek   rune
	dico   map[string]*G.Node
	g      *G.ExprGraph
	result *G.Node
	err    error
}

// Let assigns insert a node into de dictionary represented by the identifier
func (x *exprLex) Let(ident string, value *G.Node) {
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
L:
	for {
		c = x.next()
		switch {
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
L:
	for {
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

//line yacctab:1
var gorgoniaExca = [...]int{
	-1, 1,
	1, -1,
	-2, 0,
}

const gorgoniaPrivate = 57344

const gorgoniaLast = 40

var gorgoniaAct = [...]int{

	7, 9, 6, 30, 11, 12, 13, 8, 18, 1,
	19, 20, 22, 23, 24, 10, 3, 25, 26, 27,
	28, 29, 4, 14, 5, 15, 0, 9, 0, 2,
	11, 12, 13, 8, 16, 17, 0, 0, 0, 21,
}
var gorgoniaPact = [...]int{

	18, -1000, -1000, 19, 18, 18, 3, -1000, -1000, 18,
	-1000, -8, -8, -8, -8, -8, -1000, -1000, -8, -8,
	-8, -7, -1000, -1000, -1000, 3, 3, -1000, -1000, -1000,
	-1000,
}
var gorgoniaPgo = [...]int{

	0, 29, 16, 2, 0, 15, 9,
}
var gorgoniaR1 = [...]int{

	0, 6, 1, 1, 1, 2, 2, 2, 3, 3,
	3, 3, 4, 4, 4, 5, 5, 5,
}
var gorgoniaR2 = [...]int{

	0, 1, 1, 2, 2, 1, 3, 3, 1, 3,
	3, 3, 1, 3, 1, 2, 2, 2,
}
var gorgoniaChk = [...]int{

	-1000, -6, -1, -2, 4, 6, -3, -4, 15, 9,
	-5, 12, 13, 14, 4, 6, -1, -1, 5, 7,
	8, -1, -4, -4, -4, -3, -3, -4, -4, -4,
	10,
}
var gorgoniaDef = [...]int{

	0, -2, 1, 2, 0, 0, 5, 8, 12, 0,
	14, 0, 0, 0, 0, 0, 3, 4, 0, 0,
	0, 0, 15, 16, 17, 6, 7, 9, 10, 11,
	13,
}
var gorgoniaTok1 = [...]int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	9, 10, 7, 4, 3, 6, 3, 8, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 11, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 5,
}
var gorgoniaTok2 = [...]int{

	2, 3, 13, 14, 15,
}
var gorgoniaTok3 = [...]int{
	963, 12, 0,
}

var gorgoniaErrorMessages = [...]struct {
	state int
	token int
	msg   string
}{}

//line yaccpar:1

/*	parser for yacc output	*/

var (
	gorgoniaDebug        = 0
	gorgoniaErrorVerbose = false
)

type gorgoniaLexer interface {
	Lex(lval *gorgoniaSymType) int
	Error(s string)
}

type gorgoniaParser interface {
	Parse(gorgoniaLexer) int
	Lookahead() int
}

type gorgoniaParserImpl struct {
	lval  gorgoniaSymType
	stack [gorgoniaInitialStackSize]gorgoniaSymType
	char  int
}

func (p *gorgoniaParserImpl) Lookahead() int {
	return p.char
}

func gorgoniaNewParser() gorgoniaParser {
	return &gorgoniaParserImpl{}
}

const gorgoniaFlag = -1000

func gorgoniaTokname(c int) string {
	if c >= 1 && c-1 < len(gorgoniaToknames) {
		if gorgoniaToknames[c-1] != "" {
			return gorgoniaToknames[c-1]
		}
	}
	return __yyfmt__.Sprintf("tok-%v", c)
}

func gorgoniaStatname(s int) string {
	if s >= 0 && s < len(gorgoniaStatenames) {
		if gorgoniaStatenames[s] != "" {
			return gorgoniaStatenames[s]
		}
	}
	return __yyfmt__.Sprintf("state-%v", s)
}

func gorgoniaErrorMessage(state, lookAhead int) string {
	const TOKSTART = 4

	if !gorgoniaErrorVerbose {
		return "syntax error"
	}

	for _, e := range gorgoniaErrorMessages {
		if e.state == state && e.token == lookAhead {
			return "syntax error: " + e.msg
		}
	}

	res := "syntax error: unexpected " + gorgoniaTokname(lookAhead)

	// To match Bison, suggest at most four expected tokens.
	expected := make([]int, 0, 4)

	// Look for shiftable tokens.
	base := gorgoniaPact[state]
	for tok := TOKSTART; tok-1 < len(gorgoniaToknames); tok++ {
		if n := base + tok; n >= 0 && n < gorgoniaLast && gorgoniaChk[gorgoniaAct[n]] == tok {
			if len(expected) == cap(expected) {
				return res
			}
			expected = append(expected, tok)
		}
	}

	if gorgoniaDef[state] == -2 {
		i := 0
		for gorgoniaExca[i] != -1 || gorgoniaExca[i+1] != state {
			i += 2
		}

		// Look for tokens that we accept or reduce.
		for i += 2; gorgoniaExca[i] >= 0; i += 2 {
			tok := gorgoniaExca[i]
			if tok < TOKSTART || gorgoniaExca[i+1] == 0 {
				continue
			}
			if len(expected) == cap(expected) {
				return res
			}
			expected = append(expected, tok)
		}

		// If the default action is to accept or reduce, give up.
		if gorgoniaExca[i+1] != 0 {
			return res
		}
	}

	for i, tok := range expected {
		if i == 0 {
			res += ", expecting "
		} else {
			res += " or "
		}
		res += gorgoniaTokname(tok)
	}
	return res
}

func gorgonialex1(lex gorgoniaLexer, lval *gorgoniaSymType) (char, token int) {
	token = 0
	char = lex.Lex(lval)
	if char <= 0 {
		token = gorgoniaTok1[0]
		goto out
	}
	if char < len(gorgoniaTok1) {
		token = gorgoniaTok1[char]
		goto out
	}
	if char >= gorgoniaPrivate {
		if char < gorgoniaPrivate+len(gorgoniaTok2) {
			token = gorgoniaTok2[char-gorgoniaPrivate]
			goto out
		}
	}
	for i := 0; i < len(gorgoniaTok3); i += 2 {
		token = gorgoniaTok3[i+0]
		if token == char {
			token = gorgoniaTok3[i+1]
			goto out
		}
	}

out:
	if token == 0 {
		token = gorgoniaTok2[1] /* unknown char */
	}
	if gorgoniaDebug >= 3 {
		__yyfmt__.Printf("lex %s(%d)\n", gorgoniaTokname(token), uint(char))
	}
	return char, token
}

func gorgoniaParse(gorgonialex gorgoniaLexer) int {
	return gorgoniaNewParser().Parse(gorgonialex)
}

func (gorgoniarcvr *gorgoniaParserImpl) Parse(gorgonialex gorgoniaLexer) int {
	var gorgonian int
	var gorgoniaVAL gorgoniaSymType
	var gorgoniaDollar []gorgoniaSymType
	_ = gorgoniaDollar // silence set and not used
	gorgoniaS := gorgoniarcvr.stack[:]

	Nerrs := 0   /* number of errors */
	Errflag := 0 /* error recovery flag */
	gorgoniastate := 0
	gorgoniarcvr.char = -1
	gorgoniatoken := -1 // gorgoniarcvr.char translated into internal numbering
	defer func() {
		// Make sure we report no lookahead when not parsing.
		gorgoniastate = -1
		gorgoniarcvr.char = -1
		gorgoniatoken = -1
	}()
	gorgoniap := -1
	goto gorgoniastack

ret0:
	return 0

ret1:
	return 1

gorgoniastack:
	/* put a state and value onto the stack */
	if gorgoniaDebug >= 4 {
		__yyfmt__.Printf("char %v in %v\n", gorgoniaTokname(gorgoniatoken), gorgoniaStatname(gorgoniastate))
	}

	gorgoniap++
	if gorgoniap >= len(gorgoniaS) {
		nyys := make([]gorgoniaSymType, len(gorgoniaS)*2)
		copy(nyys, gorgoniaS)
		gorgoniaS = nyys
	}
	gorgoniaS[gorgoniap] = gorgoniaVAL
	gorgoniaS[gorgoniap].yys = gorgoniastate

gorgonianewstate:
	gorgonian = gorgoniaPact[gorgoniastate]
	if gorgonian <= gorgoniaFlag {
		goto gorgoniadefault /* simple state */
	}
	if gorgoniarcvr.char < 0 {
		gorgoniarcvr.char, gorgoniatoken = gorgonialex1(gorgonialex, &gorgoniarcvr.lval)
	}
	gorgonian += gorgoniatoken
	if gorgonian < 0 || gorgonian >= gorgoniaLast {
		goto gorgoniadefault
	}
	gorgonian = gorgoniaAct[gorgonian]
	if gorgoniaChk[gorgonian] == gorgoniatoken { /* valid shift */
		gorgoniarcvr.char = -1
		gorgoniatoken = -1
		gorgoniaVAL = gorgoniarcvr.lval
		gorgoniastate = gorgonian
		if Errflag > 0 {
			Errflag--
		}
		goto gorgoniastack
	}

gorgoniadefault:
	/* default state action */
	gorgonian = gorgoniaDef[gorgoniastate]
	if gorgonian == -2 {
		if gorgoniarcvr.char < 0 {
			gorgoniarcvr.char, gorgoniatoken = gorgonialex1(gorgonialex, &gorgoniarcvr.lval)
		}

		/* look through exception table */
		xi := 0
		for {
			if gorgoniaExca[xi+0] == -1 && gorgoniaExca[xi+1] == gorgoniastate {
				break
			}
			xi += 2
		}
		for xi += 2; ; xi += 2 {
			gorgonian = gorgoniaExca[xi+0]
			if gorgonian < 0 || gorgonian == gorgoniatoken {
				break
			}
		}
		gorgonian = gorgoniaExca[xi+1]
		if gorgonian < 0 {
			goto ret0
		}
	}
	if gorgonian == 0 {
		/* error ... attempt to resume parsing */
		switch Errflag {
		case 0: /* brand new error */
			gorgonialex.Error(gorgoniaErrorMessage(gorgoniastate, gorgoniatoken))
			Nerrs++
			if gorgoniaDebug >= 1 {
				__yyfmt__.Printf("%s", gorgoniaStatname(gorgoniastate))
				__yyfmt__.Printf(" saw %s\n", gorgoniaTokname(gorgoniatoken))
			}
			fallthrough

		case 1, 2: /* incompletely recovered error ... try again */
			Errflag = 3

			/* find a state where "error" is a legal shift action */
			for gorgoniap >= 0 {
				gorgonian = gorgoniaPact[gorgoniaS[gorgoniap].yys] + gorgoniaErrCode
				if gorgonian >= 0 && gorgonian < gorgoniaLast {
					gorgoniastate = gorgoniaAct[gorgonian] /* simulate a shift of "error" */
					if gorgoniaChk[gorgoniastate] == gorgoniaErrCode {
						goto gorgoniastack
					}
				}

				/* the current p has no shift on "error", pop stack */
				if gorgoniaDebug >= 2 {
					__yyfmt__.Printf("error recovery pops state %d\n", gorgoniaS[gorgoniap].yys)
				}
				gorgoniap--
			}
			/* there is no state on the stack with an error shift ... abort */
			goto ret1

		case 3: /* no shift yet; clobber input char */
			if gorgoniaDebug >= 2 {
				__yyfmt__.Printf("error recovery discards %s\n", gorgoniaTokname(gorgoniatoken))
			}
			if gorgoniatoken == gorgoniaEofCode {
				goto ret1
			}
			gorgoniarcvr.char = -1
			gorgoniatoken = -1
			goto gorgonianewstate /* try again in the same state */
		}
	}

	/* reduction by production gorgonian */
	if gorgoniaDebug >= 2 {
		__yyfmt__.Printf("reduce %v in:\n\t%v\n", gorgonian, gorgoniaStatname(gorgoniastate))
	}

	gorgoniant := gorgonian
	gorgoniapt := gorgoniap
	_ = gorgoniapt // guard against "declared and not used"

	gorgoniap -= gorgoniaR2[gorgonian]
	// gorgoniap is now the index of $0. Perform the default action. Iff the
	// reduced production is ε, $1 is possibly out of range.
	if gorgoniap+1 >= len(gorgoniaS) {
		nyys := make([]gorgoniaSymType, len(gorgoniaS)*2)
		copy(nyys, gorgoniaS)
		gorgoniaS = nyys
	}
	gorgoniaVAL = gorgoniaS[gorgoniap+1]

	/* consult goto table to find next state */
	gorgonian = gorgoniaR1[gorgonian]
	gorgoniag := gorgoniaPgo[gorgonian]
	gorgoniaj := gorgoniag + gorgoniaS[gorgoniap].yys + 1

	if gorgoniaj >= gorgoniaLast {
		gorgoniastate = gorgoniaAct[gorgoniag]
	} else {
		gorgoniastate = gorgoniaAct[gorgoniaj]
		if gorgoniaChk[gorgoniastate] != -gorgonian {
			gorgoniastate = gorgoniaAct[gorgoniag]
		}
	}
	// dummy call; replaced with literal code
	switch gorgoniant {

	case 1:
		gorgoniaDollar = gorgoniaS[gorgoniapt-1 : gorgoniapt+1]
		//line expr.y:35
		{
			gorgonialex.(*exprLex).result = gorgoniaDollar[1].node

		}
	case 3:
		gorgoniaDollar = gorgoniaS[gorgoniapt-2 : gorgoniapt+1]
		//line expr.y:43
		{
			gorgoniaVAL.node = gorgoniaDollar[2].node
		}
	case 4:
		gorgoniaDollar = gorgoniaS[gorgoniapt-2 : gorgoniapt+1]
		//line expr.y:47
		{
			gorgoniaVAL.node = G.Must(G.Neg(gorgoniaDollar[2].node))
		}
	case 6:
		gorgoniaDollar = gorgoniaS[gorgoniapt-3 : gorgoniapt+1]
		//line expr.y:54
		{
			gorgoniaVAL.node = G.Must(G.Add(gorgoniaDollar[1].node, gorgoniaDollar[3].node))
		}
	case 7:
		gorgoniaDollar = gorgoniaS[gorgoniapt-3 : gorgoniapt+1]
		//line expr.y:58
		{
			gorgoniaVAL.node = G.Must(G.Sub(gorgoniaDollar[1].node, gorgoniaDollar[3].node))
		}
	case 9:
		gorgoniaDollar = gorgoniaS[gorgoniapt-3 : gorgoniapt+1]
		//line expr.y:65
		{
			gorgoniaVAL.node = G.Must(G.Mul(gorgoniaDollar[1].node, gorgoniaDollar[3].node))
		}
	case 10:
		gorgoniaDollar = gorgoniaS[gorgoniapt-3 : gorgoniapt+1]
		//line expr.y:69
		{
			gorgoniaVAL.node = G.Must(G.HadamardProd(gorgoniaDollar[1].node, gorgoniaDollar[3].node))
		}
	case 11:
		gorgoniaDollar = gorgoniaS[gorgoniapt-3 : gorgoniapt+1]
		//line expr.y:73
		{
			gorgoniaVAL.node = G.Must(G.Div(gorgoniaDollar[1].node, gorgoniaDollar[3].node))
		}
	case 13:
		gorgoniaDollar = gorgoniaS[gorgoniapt-3 : gorgoniapt+1]
		//line expr.y:80
		{
			gorgoniaVAL.node = gorgoniaDollar[2].node
		}
	case 15:
		gorgoniaDollar = gorgoniaS[gorgoniapt-2 : gorgoniapt+1]
		//line expr.y:87
		{
			gorgoniaVAL.node = G.Must(G.Sigmoid(gorgoniaDollar[2].node))
		}
	case 16:
		gorgoniaDollar = gorgoniaS[gorgoniapt-2 : gorgoniapt+1]
		//line expr.y:91
		{
			gorgoniaVAL.node = G.Must(G.Tanh(gorgoniaDollar[2].node))
		}
	case 17:
		gorgoniaDollar = gorgoniaS[gorgoniapt-2 : gorgoniapt+1]
		//line expr.y:95
		{
			gorgoniaVAL.node = G.Must(G.SoftMax(gorgoniaDollar[2].node))
		}
	}
	goto gorgoniastack /* stack new state and value */
}
