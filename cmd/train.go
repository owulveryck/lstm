package main

import (
	"context"
	"io"
)

func train(ctx context.Context, network *LSTM, r io.ReadSeeker, total int) {
	/*
		maxEpoch := 1
		//miniBatchSize := 10
		var err error
		//y, err := gorgonia.StableSoftMax(network.Ht)
		if err != nil {
			log.Fatal(err)
		}
		y := gorgonia.NewVector(network.G, float, gorgonia.WithName("yₜ+1"),
			gorgonia.WithShape(network.VectorSize))
		yT := tensor.NewDense(float,
			y.Shape(),
			tensor.WithBacking(gorgonia.Ones()(float, y.Shape()...)),
		)
		gorgonia.Let(y, yT)
		vm := gorgonia.NewTapeMachine(network.G)

		for epoch := 0; epoch < maxEpoch; epoch++ {
			r.Seek(0, io.SeekStart)
			buf := bufio.NewReader(r)
			rn, _, err := buf.ReadRune()
			if err != nil {
				if err == io.EOF {
					break
				}
				log.Fatal(err)
			}
			setValue(network.Dict[rn], network.Xt.Value().Data().([]float64))

			i := 0
			bar := pb.New(total)
			// show percents (by default already true)
			bar.ShowPercent = true

			// show bar (by default already true)
			bar.ShowBar = true

			bar.ShowCounters = true

			bar.ShowTimeLeft = true
			// and start
			bar.Start()
			for {
				rn, _, err := buf.ReadRune()
				//fmt.Printf("[epoch %v] => %2.2f%%\r", epoch, float64(i)*100/float64(total))
				bar.Increment()
				if err != nil {
					if err == io.EOF {
						break
					}
					log.Fatal(err)
				}
				setValue(network.Dict[rn], y.Value().Data().([]float64))
				if network.Ht.Value() != nil {
					copy(network.Htprev.Value().Data().([]float64), network.Ht.Value().Data().([]float64))
				}
				if network.Ct.Value() != nil {
					copy(network.Ctprev.Value().Data().([]float64), network.Ct.Value().Data().([]float64))
				}
				//gorgonia.Let(network.Htprev, network.Ht.Value())
				//gorgonia.Let(network.Ctprev, network.Ct.Value())
				err = vm.RunAll()
				if err != nil {
					log.Fatal(err)
				}
				vm.Reset()

				setValue(network.Dict[rn], network.Xt.Value().Data().([]float64))
				i++
			}
		}
	*/
}
