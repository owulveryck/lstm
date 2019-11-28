package main

import (
	"flag"
	"log"
	"os"

	"github.com/owulveryck/lstm"
)

func main() {
	var inputFile, outputFile string
	flag.StringVar(&inputFile, "i", "", "input file")
	flag.StringVar(&outputFile, "o", "backup.bin", "output file")
	flag.Parse()
	if inputFile == "" {
		flag.Usage()
		os.Exit(1)
	}

	f, err := os.Open(inputFile)
	if err != nil {
		log.Fatal(err)
	}
	dict := getVocabulary(f)
	defer f.Close()
	vectorSize := len(dict)
	hiddenSize := 1024

	nn := lstm.NewLSTM(vectorSize, hiddenSize)
	nn.Dict = dict
	initLearnables(nn.Learnables())
}

func save(nn *lstm.LSTM, outputFile string) error {
	backup, err := os.Create(outputFile)
	if err != nil {
		return err
	}
	defer backup.Close()
	return nn.Save(backup)
}
