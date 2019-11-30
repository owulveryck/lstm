package main

import (
	"bufio"
	"io"
	"log"
	"sort"
)

func getVocabulary(r io.Reader) []rune {
	dict := make(map[rune]int)
	buf := bufio.NewReader(r)
	for {
		rn, _, err := buf.ReadRune()
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Fatal(err)
		}
		dict[rn] = 0
	}
	output := make([]rune, 0, len(dict))
	for rn, _ := range dict {
		output = append(output, rn)
	}

	sort.Slice(output, func(i, j int) bool {
		return output[i] > output[j]
	})
	return output
}
