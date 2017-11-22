package char

import (
	"errors"
	"io"
)

// Encoder write characters to an output stream
type Encoder struct {
	vocab *Vocab
	w     io.Writer
}

// NewEncoder returns a new encoder that writes to w
func NewEncoder(w io.Writer, vocab *Vocab) *Encoder {
	return &Encoder{
		vocab: vocab,
		w:     w,
	}
}

// Encode the 1-of-K encoded values to the stream
func (enc *Encoder) Encode(values []int) error {
	if enc.vocab == nil {
		return errors.New("Empty vocab")
	}
	for _, v := range values {
		_, err := enc.w.Write([]byte(string(enc.vocab.iToR[v])))
		if err != nil {
			return err
		}
	}
	return nil
}
