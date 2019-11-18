package main

import (
	"context"
	"flag"
	"io"
	"log"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var float = tensor.Float64

func main() {
	train := flag.Bool("train", false, "training mode")
	dataset := flag.String("dataset", "./lstm.go", "dataset")
	flag.Parse()

	if *train {
		runTrain(*dataset)
	} else {
		runPredict()
	}
}

func runTrain(dataset string) {
	f, err := os.Open(dataset)
	if err != nil {
		log.Fatal(err)
	}
	dict, total := getVocabulary(f)
	defer f.Close()
	vectorSize := len(dict)
	hiddenSize := 1024

	lstm := newLSTM(vectorSize, hiddenSize)
	lstm.Dict = dict
	initLearnables(lstm.learnableNodes(), gorgonia.Gaussian(0, 0.08))
	backup, err := os.Create("backup.bin")
	if err != nil {
		log.Fatal(err)
	}
	defer backup.Close()
	train(context.Background(), lstm, f, total)
	lstm.Save(backup)
}

func runPredict() {
	f, err := os.Open("backup.bin")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	network, err := newTrainedLSTM(f)
	if err != nil {
		log.Fatal(err)
	}
	_ = network
	f.Seek(0, io.SeekStart)
	/*
		var ht, ct, yt gorgonia.Value
		gorgonia.Read(network.Ht, &ht)
		gorgonia.Read(network.Ct, &ct)
		gorgonia.Read(network.Yt, &yt)
		vm := gorgonia.NewTapeMachine(network.G)
		for i := 0; i < 1000; i++ {
			err := vm.RunAll()
			if err != nil {
				log.Fatal(err)
			}
			gorgonia.Let(network.Htprev, ht)
			gorgonia.Let(network.Ctprev, ct)
			gorgonia.Let(network.Xt, yt)
			fmt.Printf("%s",
				string(getRune(
					network.Dict,
					getValueFromOneHot(yt.Data().([]float64)))))
			vm.Reset()
		}
	*/
}
