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

	strides := make([]int, len(shape))

	stride := 1

	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor{
		data:    make([]float64, size),
		shape:   append([]int(nil), shape...),
		strides: strides,
	}
}

func (t *Tensor) Shape() []int {
	return append([]int(nil), t.shape...)
}
