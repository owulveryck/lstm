package main

import "io"

type reader interface {
	io.ReadSeeker
	io.RuneReader
}

// move the seeker back from seekBack (in number of bytes) and goes forward of step *runes*
func move(rs reader, step int, seekBack int64) error {
	_, err := rs.Seek(-seekBack, io.SeekCurrent)
	if err != nil {
		return err
	}
	for i := 0; i < step; i++ {
		_, _, err := rs.ReadRune()
		if err != nil {
			return err
		}
	}
	return nil
}
