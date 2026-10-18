package tensor

import (
	"slices"
)

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

func (t *Tensor) Transpose(axes ...int) *Tensor {
	newShape := slices.Clone(t.shape)
	newStrides := slices.Clone(t.strides)

	if len(axes) == 0 {
		slices.Reverse(newShape)
		slices.Reverse(newStrides)

		return &Tensor{
			data:    t.data,
			shape:   newShape,
			strides: newStrides,
			offset:  t.offset,
		}
	}

	rank := t.Rank()

	if len(axes) != rank {
		panic("invalid number of axes")
	}

	seen := make([]bool, rank)

	for i, axis := range axes {
		if axis < 0 || axis >= rank {
			panic("invalid axis")
		}

		if seen[axis] {
			panic("duplicated axis")
		}

		seen[axis] = true

		newShape[i] = t.shape[axis]
		newStrides[i] = t.strides[axis]
	}

	return &Tensor{
		data:    t.data,
		shape:   newShape,
		strides: newStrides,
		offset:  t.offset,
	}
}

func (t *Tensor) Slice(start, end, axis int) *Tensor {
	if axis < 0 || axis >= t.Rank() {
		panic("invalid axis")
	}

	if start < 0 || start >= t.shape[axis] {
		panic("invalid start of slice")
	}

	if end < 0 || end > t.shape[axis] {
		panic("invalid end of slice")
	}

	if start >= end {
		panic("invalid range")
	}

	newShape := slices.Clone(t.shape)
	newShape[axis] = end - start

	newStrides := slices.Clone(t.strides)

	newOffset := t.offset + t.strides[axis]*start

	return &Tensor{
		data:    t.data,
		shape:   newShape,
		strides: newStrides,
		offset:  newOffset,
	}
}
