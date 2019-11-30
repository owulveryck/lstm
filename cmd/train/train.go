package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/owulveryck/lstm"
	"github.com/owulveryck/lstm/internal/text"
)

func train(ctx context.Context, nn *lstm.LSTM, input io.Reader, config configuration) error {
	content, err := ioutil.ReadAll(input)
	if err != nil {
		return err
	}
	fmt.Println(content)
	rdr := bytes.NewReader(content)
	for i := 0; i < config.Epoch; i++ {
		_, err := rdr.Seek(0, io.SeekStart)
		a, _, _ := rdr.ReadRune()
		fmt.Println(a)
		_, err = rdr.Seek(0, io.SeekStart)
		if err != nil {
			return err
		}
		feedC, errC := text.Feeder(ctx, nn.Dict, rdr, config.BatchSize, config.Step)

		fmt.Println(i)
		for x := range feedC {
			fmt.Println(x)
		}
		if err := <-errC; err != nil {
			if err != io.EOF {
				return err
			}
		}
	}
	return nil
}
