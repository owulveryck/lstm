package lstm

// testBackends returns a backend with predictibale values for the test
// biais are zeroes and matrices are 1
func testBackends(inputSize, outputSize int, hiddenSize int) *backends {
	var back backends
	back.InputSize = inputSize
	back.OutputSize = outputSize
	back.HiddenSize = hiddenSize
	back.Wi = make([]float32, hiddenSize*inputSize)
	for i := 0; i < hiddenSize*inputSize; i++ {
		back.Wi[i] = 1
	}
	back.Ui = make([]float32, hiddenSize*hiddenSize)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		back.Ui[i] = 1
	}
	back.BiasI = make([]float32, hiddenSize)
	back.Wo = make([]float32, hiddenSize*inputSize)
	for i := 0; i < hiddenSize*inputSize; i++ {
		back.Wo[i] = 1
	}
	back.Uo = make([]float32, hiddenSize*hiddenSize)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		back.Uo[i] = 1
	}
	back.BiasO = make([]float32, hiddenSize)
	back.Wf = make([]float32, hiddenSize*inputSize)
	for i := 0; i < hiddenSize*inputSize; i++ {
		back.Wf[i] = 1
	}
	back.Uf = make([]float32, hiddenSize*hiddenSize)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		back.Uf[i] = 1
	}
	back.BiasF = make([]float32, hiddenSize)
	back.Wc = make([]float32, hiddenSize*inputSize)
	for i := 0; i < hiddenSize*inputSize; i++ {
		back.Wc[i] = 1
	}
	back.Uc = make([]float32, hiddenSize*hiddenSize)
	for i := 0; i < hiddenSize*hiddenSize; i++ {
		back.Uc[i] = 1
	}
	back.BiasC = make([]float32, hiddenSize)
	return &back
}
