package text

import (
	"bytes"
	"io"
	"testing"
)

func TestMove(t *testing.T) {
	testinput := bytes.NewReader([]byte(`abc⌘abc⌘abc⌘abc⌘abc⌘`))
	step := 4
	i := 0
	for {
		rn, n, err := testinput.ReadRune()
		if err != nil {
			if err != io.EOF {
				t.Fatal(err)
			}
			break
		}
		if rn != 'a' {
			t.Fatal(rn)
		}
		i++
		err = move(testinput, step, int64(n))
		if err != nil {
			if err != io.EOF {
				t.Fatal(err)
			}
			break
		}
	}
	if i != 5 {
		t.Fatalf("expected 5 elements, but got %v", i)
	}
}
