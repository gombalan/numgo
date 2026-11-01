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
