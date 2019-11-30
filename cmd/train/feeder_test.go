package main

import (
	"bytes"
	"context"
	"io"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFeeder(t *testing.T) {
	config := configuration{
		Step:      5,
		BatchSize: 4,
	}
	expectedVals := []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1}
	dict := []rune{'a', 'b', 'c', 0x2318}
	testinput := bytes.NewReader([]byte(`abc⌘abc⌘abc⌘abc⌘abc⌘`))
	inputC, errC := feeder(context.Background(), dict, testinput, config)
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
		t.Log(x)
		assert.Equal(t, expectedVals, x.Data().([]float64), "data not equal")
		i++
	}
	if i != 5 {
		t.Fail()
	}
	wg.Wait()
}
