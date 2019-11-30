package text

import (
	"os"

	"github.com/owulveryck/lstm"
)

func save(nn *lstm.LSTM, outputFile string) error {
	backup, err := os.Create(outputFile)
	if err != nil {
		return err
	}
	defer backup.Close()
	return nn.Save(backup)
}
