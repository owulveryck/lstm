package lstm

import "testing"

func TestForwardStep(t *testing.T) {
	model := newModelFromBackends(testBackends(5, 5, 100))
	_, _, err := model.forwardStep(nil, model.prevHidden, model.prevCell)
	if err != nil {
		t.Fatal(err)
	}
}
