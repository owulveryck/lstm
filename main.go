package main

import (
	"io"
	"log"
	"os"

	"github.com/owulveryck/charRNN/encoding/char"
	"github.com/owulveryck/charRNN/lstm"
	"gorgonia.org/gorgonia"
)

// best function to return the best indicator
var best = func(vals []float32) int {
	best := float32(0)
	idx := 0
	for i, v := range vals {
		if v > best {
			log.Printf("%v:%v", i, v)
			best = v
			idx = i

		}
	}
	return idx
}

func main() {
	f, err := os.Open("data/sherlock/vocab.txt")
	if err != nil {
		log.Fatal(err)
	}
	vocab, err := char.NewVocab(f)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	f, err = os.Open("data/sherlock/input.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	dec := char.NewDecoder(f, nil, vocab)
	//dec.SetBreakPoint(regexp.MustCompile(regexp.QuoteMeta(`.`)))
	//var res []int
	res := make([]int, 20)
	model := lstm.NewModel(len(vocab.Letters), len(vocab.Letters), []int{500})
	//enc := char.NewEncoder(os.Stdout, vocab)

	//var l2reg = 0.0001
	var learnrate = 0.00001
	var clipVal = 5.0
	//solver := gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(learnrate), gorgonia.WithL2Reg(l2reg), gorgonia.WithClip(clipVal))
	solver := gorgonia.NewAdaGradSolver(gorgonia.WithEps(1e-8), gorgonia.WithLearnRate(learnrate), gorgonia.WithClip(clipVal))
	var cost, perp float32

	epoch := 0
	for i := 0; epoch < 100; i++ {
		//pos, _ := f.Seek(0, io.SeekCurrent)

		_, err = dec.Decode(&res)
		if err != nil && err != io.EOF {
			log.Fatal(err)
		}
		if err == io.EOF {
			epoch++
			_, err := f.Seek(0, io.SeekStart)
			if err != nil {
				log.Fatal(err)
			}
			continue
		}
		evaluate := false
		if i%100 == 0 {
			evaluate = true
		}
		cost, perp, err = model.Train(res[:len(res)-1], res[1:], evaluate, solver)
		if err != nil {
			log.Fatal(err)
		}
		if i%100 == 0 {
			log.Printf("cost: %v, perplexity: %v", cost, perp)
		}
		/*
			if i%500 == 0 {
				ctx, cancel := context.WithCancel(context.Background())
				f, err := model.Predict(ctx, []int{rand.Intn(len(vocab.Letters))}, best)
				if err != nil {
					log.Fatal(err)
				}
				n := 0
				mat := make([]int, 101)
				for v := range f {
					if n > 100 {
						cancel()
					} else {
						mat[n] = v
					}
					n++
				}
				fmt.Println("------------")
				enc.Encode(mat)
				fmt.Println("\n------------")
			}
		*/

		_, err := f.Seek(-1, io.SeekCurrent)
		if err != nil && err != io.EOF {
			log.Fatal(err)
		}

	}
}
