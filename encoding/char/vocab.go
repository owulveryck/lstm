package char

import (
	"bufio"
	"errors"
	"io"
)

// Vocab represent the vocabulary used to encode/decode a text
type Vocab struct {
	Letters []rune
	rToI    map[rune]int
	iToR    map[int]rune
}

// MarshalBinary for backup
func (v Vocab) MarshalBinary() ([]byte, error) {
	// A simple encoding: plain text.
	return []byte(string(v.Letters)), nil
}

// UnmarshalBinary for restore
func (v *Vocab) UnmarshalBinary(data []byte) error {
	v.Letters = []rune(string(data))
	return v.initMaps()
}

func (v *Vocab) initMaps() error {
	if len(v.Letters) == 0 {
		return errors.New("Vocab is empty")
	}
	v.rToI = make(map[rune]int)
	v.iToR = make(map[int]rune)
	for i, val := range v.Letters {
		v.rToI[val] = i
		v.iToR[i] = val
	}
	if len(v.Letters) != len(v.rToI) {
		return errors.New("Non uniq characters found in the vocabulary")
	}
	return nil
}

// NewVocab creates a new vocabulary from the io.Reader
func NewVocab(r io.Reader) (*Vocab, error) {
	buf := bufio.NewReader(r)
	u := make([]rune, 0)
	m := make(map[rune]bool)

	var err error
	for {
		var val rune
		val, _, err = buf.ReadRune()
		if err != nil {
			break
		}
		if _, ok := m[val]; !ok {
			m[val] = true
			u = append(u, val)
		}
	}
	v := &Vocab{
		Letters: u,
	}
	err = v.initMaps()
	return v, err
}
