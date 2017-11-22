package char

import (
	"bytes"
	"io"
	"regexp"
	"testing"
)

func TestDecode(t *testing.T) {
	vocab, err := NewVocab(bytes.NewBufferString(testData))
	if err != nil {
		t.Fatal(err)
	}
	r := bytes.NewReader([]byte(testData))
	dec := NewDecoder(r, nil, vocab)
	// Read everything
	var res []int
	_, err = dec.Decode(&res)
	if err != nil && err != io.EOF {
		t.Fatal(err)
	}
	var resultat []rune
	for _, idx := range res {
		resultat = append(resultat, vocab.iToR[idx])
	}
	if string(resultat) != testData {
		t.Fail()
	}
	dec.SetBreakPoint(regexp.MustCompile(regexp.QuoteMeta(`.`)))
	r.Seek(0, 0)
	n, err := dec.Decode(&res)
	if err != nil && err != io.EOF {
		t.Fatal(err)
	}
	if n != 56 {
		t.Fail()
	}
	// Test with a buffer size
	res = make([]int, 40)
	r.Seek(0, 0)
	n, err = dec.Decode(&res)
	if err != nil && err != io.EOF {
		t.Fatal(err)
	}
	if n != 40 {
		t.Fail()
	}
	// Buffer is bigger than the sentence until the first breakpoint
	res = make([]int, 60)
	r.Seek(0, 0)
	n, err = dec.Decode(&res)
	if err != nil && err != io.EOF {
		t.Fatal(err)
	}
	if n != 56 {
		t.Fail()
	}
	dec.vocab = nil
	_, err = dec.Decode(&res)
	if err == nil {
		t.Fail()
	}
}
