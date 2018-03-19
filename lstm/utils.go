package lstm

import (
	"strconv"
	"strings"
)

// replace is stupid function to replace the value of subscript and subscript-1 with an integer
// caution: no error checking is performed
func replace(subscript string, value int) *strings.Replacer {
	script := strings.NewReplacer(
		`1`, `₁`,
		`2`, `₂`,
		`3`, `₃`,
		`4`, `₄`,
		`5`, `₅`,
		`6`, `₆`,
		`7`, `₇`,
		`8`, `₈`,
		`9`, `₉`,
		`0`, `₀`,
		`-`, `₋`)
	r := strings.NewReplacer(subscript+`₋₁`, script.Replace(strconv.Itoa(value-1)), subscript, script.Replace(strconv.Itoa(value)))
	return r
}
