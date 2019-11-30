package text

import (
	"bytes"
	"io"
)

// move the seeker back from seekBack (in number of bytes) and goes forward of step *runes*
func move(rdr *bytes.Reader, step int, seekBack int64) error {
	_, err := rdr.Seek(-seekBack, io.SeekCurrent)
	if err != nil {
		return err
	}
	for i := 0; i < step; i++ {
		_, _, err := rdr.ReadRune()
		if err != nil {
			return err
		}
	}
	return nil
}
