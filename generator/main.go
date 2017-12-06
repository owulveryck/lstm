// this reads the stdin until EOF and output a list of all characters used
package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
)

func main() {
	vocab := make(map[rune]struct{}, 0)
	r := bufio.NewReader(os.Stdin)
	for {
		if c, _, err := r.ReadRune(); err != nil {
			if err == io.EOF {
				// Restart the training if it's not the last epoch
				for v := range vocab {
					fmt.Printf("%c", v)
				}
				return
			}
			log.Fatal(err)
		} else {
			vocab[c] = struct{}{}
		}
	}
}
