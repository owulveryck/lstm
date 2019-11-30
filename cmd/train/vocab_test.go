package main

import (
	"bytes"
	"testing"
)

func TestGetVocabulary(t *testing.T) {
	input := bytes.NewReader([]byte(`abcdéabcdéá`))
	dict := getVocabulary(input)
	if string(dict) != `éádcba` {
		t.Fatal(dict)
	}
}
