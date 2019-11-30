package main

import (
	"flag"
	"log"
	"os"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/lstm"
)

type configuration struct {
	HiddenSize int `envconfig:"HIDDEN_SIZE" default:"1024" required:"true"`
	Epoch      int `envconfig:"EPOCH" default:"100" required:"true"`
	BatchSize  int `envconfig:"BATCH_SIZE" default:"100" required:"true"`
	Step       int `envconfig:"STEP" default:"1" required:"true"`
}

func usage() error {
	flag.Usage()
	var config configuration
	return envconfig.Usage("lstm", &config)
}

func main() {
	var inputFile, outputFile string
	var help bool
	var config configuration
	flag.StringVar(&inputFile, "i", "", "input file")
	flag.StringVar(&outputFile, "o", "backup.bin", "output file")
	flag.BoolVar(&help, "h", false, "help")
	flag.Parse()
	if help || inputFile == "" {
		usage()
		return
	}

	err := envconfig.Process("lstm", &config)
	if err != nil {
		log.Fatal(err.Error())
	}

	f, err := os.Open(inputFile)
	if err != nil {
		log.Fatal(err)
	}
	dict := getVocabulary(f)
	defer f.Close()
	vectorSize := len(dict)

	nn := lstm.NewLSTM(vectorSize, config.HiddenSize)
	nn.Dict = dict
	initLearnables(nn.Learnables())
}
