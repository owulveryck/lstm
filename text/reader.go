package text

import (
	"bufio"
	"io"
)

// move the seeker back from seekBack (in number of bytes) and goes forward of step *runes*
func move(rs io.ReadSeeker, step int, seekBack int64) error {
	buf := bufio.NewReader(rs)
	_, err := rs.Seek(-seekBack, io.SeekCurrent)
	if err != nil {
		return err
	}
	for i := 0; i < step; i++ {
		_, _, err := buf.ReadRune()
		if err != nil {
			return err
		}
	}
	return nil
}
