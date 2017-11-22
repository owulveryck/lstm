package char

import (
	"bufio"
	"bytes"
	"testing"
)

func TestEncode(t *testing.T) {
	vocab, err := NewVocab(bytes.NewBufferString(testData))
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	enc := NewEncoder(writer, vocab)
	err = enc.Encode(testMatrix)
	if err != nil {
		t.Fatal(err)
	}
	err = writer.Flush()
	if err != nil {
		t.Fatal(err)
	}
	if b.String() != testData {
		t.Fail()
	}
	err = enc.Encode([]int{})
	if err != nil {
		t.Fatal(err)
	}
	enc.vocab = nil
	err = enc.Encode(testMatrix)
	if err == nil {
		t.Fail()
	}

}
