package main

import (
	"bufio"
	"io"
	"log"
)

func getRune(dict map[rune]int, idx int) rune {
	for r, v := range dict {
		if v == idx {
			return r
		}
	}
	return 0
}

func setValue(idx int, dest []float64) {
	for i := 0; i < len(dest); i++ {
		if i == idx {
			dest[i] = 1
		} else {
			dest[i] = 0
		}
	}
}

func getValueFromOneHot(f []float64) int {
	if f == nil || len(f) == 0 {
		log.Fatal("cannot get value from an empty one-hot vector")
	}
	if len(f) == 1 {
		return 0
	}
	val := 0
	max := f[0]
	for i := 1; i < len(f); i++ {
		if f[i] > max {
			val = i
		}
	}
	return val
}

func getVocabulary(r io.Reader) (map[rune]int, int) {
	dict := make(map[rune]int)
	buf := bufio.NewReader(r)
	total := 0
	for {
		rn, _, err := buf.ReadRune()
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Fatal(err)
		}
		total++
		dict[rn] = 0
	}
	i := 0
	for rn, _ := range dict {
		dict[rn] = i
		i++
	}
	return dict, total
}
