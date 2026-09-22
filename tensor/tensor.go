package tensor

type Tensor struct {
	data    []float64
	shape   []int
	strides []int
}

func New(shape ...int) *Tensor {
	if len(shape) == 0 {
		panic("tensor must have at least one dimension")
	}

	size := 1

	for _, n := range shape {
		if n <= 0 {
			panic("invalid shape")
		}

		size *= n
	}

	return &Tensor{
		data:    make([]float64, size),
		shape:   append([]int(nil), shape...),
		strides: makeStrides(shape...),
	}
}

func makeStrides(shape ...int) []int {
	strides := make([]int, len(shape))

	stride := 1

	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return strides
}

func (t *Tensor) Shape() []int {
	return append([]int(nil), t.shape...)
}

func (t *Tensor) Rank() int {
	return len(t.shape)
}

func (t *Tensor) Size() int {
	return len(t.data)
}

func (t *Tensor) Reshape(shape ...int) *Tensor {
	if len(shape) == 0 {
		panic("tensor must have at least one dimension")
	}

	size := 1

	for _, n := range shape {
		if n <= 0 {
			panic("invalid shape")
		}

		size *= n
	}

	if size != len(t.data) {
		panic("reshape size mismatch")
	}

	return &Tensor{
		data:    t.data,
		shape:   append([]int(nil), shape...),
		strides: makeStrides(shape...),
	}
}
