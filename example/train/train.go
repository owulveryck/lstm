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
	runes = `;HFrch.vG
dMgEARDKt:q$sV-Px&jzel?I!mkSyWNnB,LiaOUJfbuQwY'ZXCop3T `
	filename = "../../data/shakespeare/input.txt"
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
	vocabSize := len([]rune(runes))
	model := lstm.NewModel(vocabSize, vocabSize, 100)
	learnrate := 1e-3
	l2reg := 1e-6
	clipVal := float64(5)
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))

	for i := 0; i < 100; i++ {
		f, err := os.Open(filename)
		if err != nil {
			log.Fatal(err)
		}
		tset := char.NewTrainingSet(f, runeToIdx, vocabSize, 25, 1)
		pause := make(chan struct{})
		infoChan, errc := model.Train(context.TODO(), tset, solver, pause)
		iter := 1
		for infos := range infoChan {
			if iter%100 == 0 {
				fmt.Printf("%v\n", infos)
			}
			if iter%500 == 0 {
				fmt.Println("\nGoing to predict")
				pause <- struct{}{}
				prediction := char.NewPrediction("Hello, ", runeToIdx, 500, vocabSize)
				err := model.Predict(context.TODO(), prediction)
				if err != nil {
					log.Println(err)
					continue
				}

				for _, output := range prediction.GetOutput() {
					var idx int
					for i, val := range output {
						if val == 1 {
							idx = i
						}
					}
					rne, err := idxToRune(idx)
					if err != nil {
						log.Fatal(err)
					}
					fmt.Printf(string(rne))
				}
				fmt.Println("")
				pause <- struct{}{}
			}
			iter++
		}
		err = <-errc
		if err == io.EOF {
			close(pause)
			return
		}
		if err != nil && err != io.EOF {
			log.Fatal(err)
		}
		f.Close()
	}

}
