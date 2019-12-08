package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/owulveryck/lstm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	f, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	nn, err := lstm.NewTrainedLSTM(f)
	if err != nil {
		log.Fatal(err)
	}
	model := lstm.NewNetwork(nn, 1)
	log.Println(len(model.Y))
	shape := model.H[0].Shape()
	hT := tensor.NewDense(tensor.Float64, shape, tensor.WithBacking(make([]float64, shape[0])))
	gorgonia.Let(model.H[0], hT)
	shape = model.C[0].Shape()
	cT := tensor.NewDense(tensor.Float64, shape, tensor.WithBacking(make([]float64, shape[0])))
	gorgonia.Let(model.C[0], cT)
	vm := gorgonia.NewTapeMachine(nn.G)

	for i := 0; i < len(nn.Dict); i++ {
		fmt.Printf("%c", nn.Dict[i])
	}
	fmt.Printf("\n")
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("Enter text: ")
		line, _ := reader.ReadSlice('\n')
		rdr := bytes.NewReader(line[:len(line)-1])
		var err error
		for {
			var rn rune
			rn, _, err = rdr.ReadRune()
			if err != nil {
				break
			}
			idx, ok := inDict(rn, nn.Dict)
			if !ok {
				fmt.Printf("Invalid rune %c\n", rn)
				continue
			}
			backend := make([]float64, len(nn.Dict))
			backend[idx] = 1
			xT := tensor.NewDense(tensor.Float64, []int{model.X[0].Shape()[0], 1}, tensor.WithBacking(backend))
			gorgonia.Let(model.X[0], xT)
			err = vm.RunAll()
			if err != nil {
				log.Fatal(err)
			}
			gorgonia.Let(model.H[0], model.H[1].Value())
			gorgonia.Let(model.C[0], model.C[1].Value())
			vm.Reset()
		}
		gorgonia.Let(model.H[0], model.H[1].Value())
		gorgonia.Let(model.C[0], model.C[1].Value())
		vm.Reset()
		for i := 0; i < 25; i++ {
			idx := getIdx(model.Y[0].Value().Data().([]float64))
			backend := make([]float64, len(nn.Dict))
			backend[idx] = 1
			xT := tensor.NewDense(tensor.Float64, []int{model.X[0].Shape()[0], 1}, tensor.WithBacking(backend))
			gorgonia.Let(model.X[0], xT)
			err = vm.RunAll()
			if err != nil {
				log.Fatal(err)
			}
			gorgonia.Let(model.H[0], model.H[1].Value())
			gorgonia.Let(model.C[0], model.C[1].Value())
			fmt.Printf("%c", getRune(model.Y[0].Value().Data().([]float64), nn.Dict))
			vm.Reset()
		}
		fmt.Printf("\n")

		if err != nil && err != io.EOF {
			log.Fatal(err)
		}
		_ = line
	}
}

func inDict(r rune, dict []rune) (int, bool) {
	for i := 0; i < len(dict); i++ {
		if r == dict[i] {
			return i, true
		}
	}
	return 0, false
}

func getRune(a []float64, dict []rune) rune {
	max := 0.0
	var rn rune
	for i := 0; i < len(a); i++ {
		if a[i] > max {
			max = a[i]
			rn = dict[i]
		}
	}
	return rn
}

func getIdx(f []float64) int {
	idx := 0
	max := 0.0
	for i := 0; i < len(f); i++ {
		if f[i] > max {
			max = f[i]
			idx = i
		}
	}
	return idx
}
