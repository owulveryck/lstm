package lstm

import (
	"encoding/gob"
	"io"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func (l *LSTM) Save(w io.Writer) error {
	enc := gob.NewEncoder(w)
	var err error
	err = enc.Encode(l.VectorSize)
	if err != nil {
		return err
	}
	err = enc.Encode(l.HiddenSize)
	if err != nil {
		return err
	}
	for i := 0; i < len(l.learnableNodes()); i++ {
		err = enc.Encode(l.learnableNodes()[i].Value())
		if err != nil {
			return err
		}
	}
	err = enc.Encode(l.Dict)
	return err
}

// Create returns a trained LSTM with values extracted from a backup
func NewTrainedLSTM(r io.Reader) (*LSTM, error) {
	dec := gob.NewDecoder(r)
	var vectorSize, hiddenSize int
	var err error
	err = dec.Decode(&vectorSize)
	if err != nil {
		return nil, err
	}
	err = dec.Decode(&hiddenSize)
	if err != nil {
		return nil, err
	}
	lstm := NewLSTM(vectorSize, hiddenSize)
	for i := 0; i < len(lstm.learnableNodes()); i++ {
		currentNode := lstm.learnableNodes()[i]
		var t tensor.Dense
		err = dec.Decode(&t)
		if err != nil {
			return nil, err
		}
		err = gorgonia.Let(currentNode, &t)
		if err != nil {
			return nil, err
		}

	}
	var dict []rune
	err = dec.Decode(&dict)
	if err != nil {
		return nil, err
	}
	lstm.Dict = dict
	return lstm, nil
}
