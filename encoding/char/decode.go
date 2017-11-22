package char

import (
	"bufio"
	"errors"
	"io"
	"regexp"
)

// Decoder reads and decode characters from the input stream
type Decoder struct {
	vocab      *Vocab
	r          io.Reader
	buffSize   int            // if n is >0, the decoder will try to decode up to n runes
	breakPoint *regexp.Regexp // if not nil, the decoder will try to code rune until the matching regexp
}

// NewDecoder returns a new decoder that reads from r.
// if breakPoint is not nil, the the decoder process the chars in the stream until a char matching the regexp is found.
func NewDecoder(r io.Reader, breakPoint *regexp.Regexp, vocab *Vocab) *Decoder {
	return &Decoder{
		breakPoint: breakPoint,
		r:          r,
		vocab:      vocab,
	}
}

// SetBreakPoint sets the breakpoint to the regexp
func (dec *Decoder) SetBreakPoint(re *regexp.Regexp) {
	dec.breakPoint = re
}

// Decode reads the next characters from its input and stores it in the value pointed to by v.
// if len(values) == 0, Decode reads until a breakPoint match, or until EOF
// it returns the number of character read and an error
func (dec *Decoder) Decode(values *[]int) (int, error) {
	var n = 0
	if dec.vocab == nil {
		return n, errors.New("Vocab not set")
	}

	buf := bufio.NewReader(dec.r)
	length := len(*values)

	var err error
	var char rune
	for n = 0; n < length || length == 0; n++ {
		char, _, err = buf.ReadRune()
		if err != nil {
			break
		}
		if length == 0 {
			*values = append(*values, dec.vocab.rToI[char])

		} else {
			(*values)[n] = dec.vocab.rToI[char]
		}

		if dec.breakPoint != nil && dec.breakPoint.MatchString(string(char)) {
			break
		}
	}
	return n, err
}
