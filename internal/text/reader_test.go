package text

import (
	"bytes"
	"io"
	"log"
	"testing"
)

func TestMove(t *testing.T) {
	testinput := bytes.NewReader([]byte(`abc⌘abc⌘abc⌘abc⌘abc⌘`))
	step := 4
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
		err = move(testinput, step, int64(n))
		if err != nil {
			log.Fatal(err)
		}
	}
}
