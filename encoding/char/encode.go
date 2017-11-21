package char

import (
	"io"
)

// Encoder write characters to an output stream
type Encoder struct {
	w io.Writer
}

// NewEncoder returns a new encoder that writes to w
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{
		w: w,
	}
}

// Encode the 1-of-K encoded values to the stream
func (enc *Encoder) Encode(values []int) error {
	return nil
}
