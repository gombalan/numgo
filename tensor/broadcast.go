package tensor

import "fmt"

func broadcastShapes(a, b []int) ([]int, error) {
	rank := max(len(a), len(b))
	shape := make([]int, rank)

	for i := range rank {
		ai := len(a) - 1 - i
		bi := len(b) - 1 - i

		aDim, bDim := 1, 1

		if ai >= 0 {
			aDim = a[ai]
		}
		if bi >= 0 {
			bDim = b[bi]
		}

		if aDim != bDim && aDim != 1 && bDim != 1 {
			return nil, fmt.Errorf("incompatible shapes: %v and %v", a, b)
		}

		shape[rank-1-i] = max(aDim, bDim)
	}

	return shape, nil
}

func broadcastStrides(shape, strides, outputShape []int) []int {
	result := make([]int, len(outputShape))

	offset := len(outputShape) - len(shape)

	for i := range outputShape {
		inputAxis := i - offset

		if inputAxis < 0 {
			result[i] = 0
			continue
		}

		if shape[inputAxis] == 1 && outputShape[i] > 1 {
			result[i] = 0
			continue
		}

		result[i] = strides[inputAxis]
	}

	return result
}
