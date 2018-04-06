package main

import (
	"bufio"
	"context"
	"encoding/gob"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/lstm"
	"github.com/owulveryck/lstm/datasetter/char"
	G "gorgonia.org/gorgonia"
)

type configuration struct {
	Dump string `envconfig:"dump" default:"/tmp/dump.bin"`
}

func newVocabulary(filename string) (vocabulary, error) {
	vocab := make(map[rune]struct{}, 0)
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	for {
		if c, _, err := r.ReadRune(); err != nil {
			if err == io.EOF {
				break
			}
			log.Fatal(err)
		} else {
			vocab[c] = struct{}{}
		}
	}
	output := make([]rune, len(vocab))
	i := 0
	for rne := range vocab {
		output[i] = rne
		i++
	}
	return output, nil

}

type vocabulary []rune

type backup struct {
	Model      lstm.Model
	Vocabulary vocabulary
}

func (v vocabulary) runeToIdx(r rune) (int, error) {
	for i := range v {
		if v[i] == r {
			return i, nil
		}
	}
	return 0, fmt.Errorf("Rune %v is not part of the vocabulary", string(r))
}

func (v vocabulary) idxToRune(i int) (rune, error) {
	var rn rune
	if i >= len(v) {
		return rn, fmt.Errorf("index invalid, no rune references")
	}
	return v[i], nil
}

func main() {
	var config configuration
	err := envconfig.Process("TRAIN", &config)
	if err != nil {
		log.Fatal(err)
	}
	filename := os.Args[1]
	vocab, err := newVocabulary(filename)
	if err != nil {
		log.Fatal(err)
	}
	vocabSize := len(vocab)
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
		max, _ := f.Seek(0, io.SeekEnd)
		f.Seek(0, io.SeekStart)
		tset := char.NewTrainingSet(f, vocab.runeToIdx, vocabSize, 30, 1)
		pause := make(chan struct{})
		infoChan, errc := model.Train(context.TODO(), tset, solver, pause)
		iter := 1
		var minLoss float32
		for infos := range infoChan {
			if iter%100 == 0 {
				if minLoss == 0 {
					minLoss = infos.Cost
				}
				if infos.Cost < minLoss {
					minLoss = infos.Cost
					log.Println("Backup because loss is minimum")
					bkp := backup{
						Model:      *model,
						Vocabulary: vocab,
					}
					f, err := os.OpenFile(config.Dump, os.O_RDWR|os.O_CREATE, 0755)
					if err != nil {
						log.Println(err)
					}
					enc := gob.NewEncoder(f)
					err = enc.Encode(bkp)
					if err != nil {
						log.Println(err)
					}
					if err := f.Close(); err != nil {
						log.Println(err)
					}
					// Do a backup
				}
				here, _ := f.Seek(0, io.SeekCurrent)
				fmt.Printf("[%v/%v]%v\n", here, max, infos)
			}
			if iter%500 == 0 {
				fmt.Println("\nGoing to predict")
				pause <- struct{}{}
				prediction := char.NewPrediction("B", vocab.runeToIdx, 100, vocabSize)
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
					rne, err := vocab.idxToRune(idx)
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
			//return
		}
		if err != nil && err != io.EOF {
			log.Fatal(err)
		}
		f.Close()
	}

}
