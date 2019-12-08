package text

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFeeder(t *testing.T) {
	batchSize := 4
	step := 4
	expectedVals := []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1}
	dict := []rune{'a', 'b', 'c', 0x2318}
	testinput := bytes.NewReader([]byte(`abc⌘abc⌘abc⌘abc⌘abc⌘`))
	inputC, errC := Feeder(context.Background(), dict, testinput, batchSize, step)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for err := range errC {
			if err != io.EOF {
				t.Fatal(err)
			}
		}
	}()
	i := 0
	for x := range inputC {
		assert.Equal(t, expectedVals, x.T.Data().([]float64), "data not equal")
		i++
	}
	if i != 5 {
		t.Fatalf("expected 5 values, but got %v", i)
	}
	wg.Wait()
}

func ExampleFeeder() {
	dict := []rune{'a', 'b', 'c', 0x2318}
	testinput := bytes.NewReader([]byte(`cba⌘`))
	inputC, _ := Feeder(context.Background(), dict, testinput, 4, 1)
	x := <-inputC
	fmt.Println(x.T)
	// Output:⎡0  0  1  0⎤
	// ⎢0  1  0  0⎥
	// ⎢1  0  0  0⎥
	// ⎣0  0  0  1⎦
}
