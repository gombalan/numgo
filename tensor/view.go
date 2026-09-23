package tensor

import "slices"

func (t *Tensor) Reshape(shape ...int) *Tensor {
	if len(shape) == 0 {
		panic("tensor must have at least one dimension")
	}

	if !t.IsContiguous() {
		panic("cannot reshape non-contiguous tensor")
	}

	newSize := 1

	for _, n := range shape {
		if n <= 0 {
			panic("invalid shape")
		}

		newSize *= n
	}

	if newSize != len(t.data) {
		panic("reshape size mismatch")
	}

	return &Tensor{
		data:    t.data,
		shape:   append([]int(nil), shape...),
		strides: makeStrides(shape...),
		offset:  t.offset,
	}
}

func (t *Tensor) Transpose() *Tensor {
	newShape := slices.Clone(t.shape)
	slices.Reverse(newShape)

	newStrides := slices.Clone(t.strides)
	slices.Reverse(newStrides)

	return &Tensor{
		data:    t.data,
		shape:   newShape,
		strides: newStrides,
		offset:  t.offset,
	}
}
