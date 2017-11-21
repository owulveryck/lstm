package char

import (
	"bytes"
	"io"
	"testing"
)

const testData = `
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam et interdum sem, ac venenatis ante. Duis ut ornare lectus. Fusce quis lectus blandit, tempor diam sit amet, porttitor felis. Maecenas vel tincidunt risus, id placerat dolor. Nullam at vehicula est. Aliquam sollicitudin libero in justo cursus aliquam. Mauris mattis sit amet magna a ultrices. Nunc nec faucibus quam. Praesent sit amet odio neque. Nam finibus luctus metus, vitae fringilla massa bibendum sed. Donec urna est, faucibus vitae congue non, mollis vitae mauris. Curabitur feugiat, orci sed feugiat varius, libero sapien iaculis enim, et auctor magna orci sit amet eros. In sodales orci eros, in ultricies massa eleifend ac. Praesent maximus turpis et nunc ullamcorper finibus.

Nunc eleifend ipsum in augue faucibus accumsan. Aenean libero sapien, interdum ac commodo non, rutrum nec velit. Praesent rhoncus, augue ac aliquam posuere, erat lorem imperdiet ex, dignissim congue justo ex non sapien. Maecenas ut vestibulum nibh. Duis in sem arcu. Nullam ultrices mattis tincidunt. Integer ullamcorper lobortis tempus. Aenean euismod sagittis lectus, sed dapibus est aliquet at. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi ut aliquam massa, a bibendum risus. Aliquam maximus leo eget tellus volutpat, vitae pellentesque dui elementum. Sed sagittis malesuada lacus, et pulvinar ipsum dignissim in. Nullam a nisi augue.
`

func TestDecode(t *testing.T) {
	vocab, err := NewVocab(bytes.NewBufferString(testData))
	if err != nil {
		t.Fatal(err)
	}
	dec := NewDecoder(bytes.NewBufferString(testData), nil, vocab)
	// Read everything
	var res []int
	_, err = dec.Decode(&res)
	if err != nil && err != io.EOF {
		t.Fatal(err)
	}
}
