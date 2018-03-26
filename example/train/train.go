package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/owulveryck/lstm"
	"github.com/owulveryck/lstm/datasetter/char"
	G "gorgonia.org/gorgonia"
)

const (
	runes = `b«ME'àésèêüivOùquHÉa-A!ÇnJjçTByVepY,?xôXCmïW wfFU(gLNQ»R:°dPDîIktrcz.Shloâ)Gû
`
	filename = "../../data/tontons/input.txt"
)

var asRunes = []rune(runes)

func runeToIdx(r rune) (int, error) {
	for i := range asRunes {
		if asRunes[i] == r {
			return i, nil
		}
	}
	return 0, fmt.Errorf("Rune %v is not part of the vocabulary", string(r))
}

func idxToRune(i int) (rune, error) {
	var rn rune
	if i >= len([]rune(runes)) {
		return rn, fmt.Errorf("index invalid, no rune references")
	}
	return []rune(runes)[i], nil
}

func main() {
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	vocabSize := len([]rune(runes))
	tset := char.NewTrainingSet(f, runeToIdx, vocabSize, 25, 1)
	model := lstm.NewModel(vocabSize, vocabSize, 100)
	learnrate := 0.01
	l2reg := 1e-6
	clipVal := float64(5)
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))

	pause := make(chan struct{})
	infoChan, errc := model.Train(context.TODO(), tset, solver, pause)
	for infos := range infoChan {
		fmt.Printf("\t\t|%v\n", infos)
	}
	err = <-errc
	if err == io.EOF {
		close(pause)
		return
	}
	if err != nil && err != io.EOF {
		log.Fatal(err)
	}

}
