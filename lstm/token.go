package lstm

// Token represents a lexical token.
type Token int

const (
	// Special tokens
	ILLEGAL Token = iota
	EOF
	WS

	// Literals
	IDENT // main

	// Keywords
	DOTPRODUCT
	PRODUCT
	ADDITION
)
