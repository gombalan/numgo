package tensor

func (t *Tensor) At(indices ...int) float64 {
	return t.data[t.offsetOf(indices...)]
}

func (t *Tensor) Set(value float64, indices ...int) {
	t.data[t.offsetOf(indices...)] = value
}

func (t *Tensor) offsetOf(indices ...int) int {
	if len(indices) != len(t.shape) {
		panic("invalid indices")
	}

	offset := t.offset

	for i := range indices {
		if indices[i] < 0 || indices[i] >= t.shape[i] {
			panic("invalid indices")
		}

		offset += indices[i] * t.strides[i]
	}

	return offset
}
