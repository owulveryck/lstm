package lstm_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/owulveryck/charRNN/lstm"
)

// Ensure the scanner can scan tokens correctly.
func TestScanner_Scan(t *testing.T) {
	var tests = []struct {
		s   string
		tok lstm.Token
		lit string
	}{
		// Special tokens (EOF, ILLEGAL, WS)
		{s: ``, tok: lstm.EOF},
		{s: `#`, tok: lstm.ILLEGAL, lit: `#`},
		{s: ` `, tok: lstm.WS, lit: " "},
		{s: "\t", tok: lstm.WS, lit: "\t"},
		{s: "\n", tok: lstm.WS, lit: "\n"},

		// Identifiers
		{s: `fₜ₋₁`, tok: lstm.IDENT, lit: `fₜ₋₁`},

		// Keywords
		{s: `·`, tok: lstm.DOTPRODUCT, lit: `·`},
		{s: `*`, tok: lstm.PRODUCT, lit: `*`},
		{s: `+`, tok: lstm.ADDITION, lit: `+`},
	}

	for i, tt := range tests {
		s := lstm.NewScanner(strings.NewReader(tt.s))
		tok, lit := s.Scan()
		if tt.tok != tok {
			t.Errorf("%d. %q token mismatch: exp=%q got=%q <%q>", i, tt.s, tt.tok, tok, lit)
		} else if tt.lit != lit {
			t.Errorf("%d. %q literal mismatch: exp=%q got=%q", i, tt.s, tt.lit, lit)
		}
	}
}

func ExampleScan() {
	s := lstm.NewScanner(strings.NewReader(`Wh·xₜ₋₁+yₐ`))
	var tok lstm.Token
	var lit string
	for tok != lstm.EOF {
		tok, lit = s.Scan()
		fmt.Printf("<%v>%v\n", tok, lit)
	}
	// Output:
	//<3>Wh
	//<4>·
	//<3>xₜ₋₁
	//<6>+
	//<3>yₐ
	//<1>
}
