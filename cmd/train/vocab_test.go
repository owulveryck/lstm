package main

import (
	"bytes"
	"testing"
)

func TestGetVocabulary(t *testing.T) {
	input := bytes.NewBufferString(`abcdéabcdéá`)
	dict := getVocabulary(input)
	if string(dict) != `éádcba` {
		t.Fatal(dict)
	}
}
