package tensor

func (t *Tensor) offsetOf(indices ...int) int {
	if len(indices) != len(t.shape) {
		panic("invalid indices")
	}

	offset := 0
	for i := 0; i < len(indices); i++ {
		if indices[i] < 0 || indices[i] >= t.shape[i] {
			panic("invalid indices")
		}

		offset += indices[i] * t.strides[i]
	}

	return offset
}
