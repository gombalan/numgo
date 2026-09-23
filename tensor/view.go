package tensor

import "slices"

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
