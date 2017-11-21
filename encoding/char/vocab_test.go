package char

import (
	"testing"
)

func TestVocabUnmarshalBinary(t *testing.T) {
	var v Vocab
	err := v.UnmarshalBinary(nil)
	if err == nil {
		t.Fail()
	}
	err = v.UnmarshalBinary([]byte("dummy"))
	if err == nil {
		t.Fail()
	}
	// Encode a nil byte array
	var input []rune
	err = v.UnmarshalBinary([]byte(string(input)))
	if err == nil {
		t.Fail()
	}
	err = v.UnmarshalBinary([]byte("abc"))
	if err != nil {
		t.Fatal(err)
	}
	err = v.UnmarshalBinary([]byte("aabc"))
	if err == nil {
		t.Fatal("aabc")
	}

	vv := Vocab{
		Letters: []rune("èô"),
	}
	b, _ := vv.MarshalBinary()
	err = v.UnmarshalBinary(b)
	if err != nil {
		t.Fatal(err)
	}
	if len(vv.Letters) != len(v.Letters) {
		t.Fatal("Backup and restore size differs, got %v, expect %v", vv.Letters, v.Letters)

	}
	for i := range vv.Letters {
		if vv.Letters[i] != v.Letters[i] {
			t.Fatal("Backup / Restored failed, expected %v, got %v", vv.Letters[i], v.Letters[i])
		}
	}
}
