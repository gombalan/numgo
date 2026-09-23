package tensor

import (
	"reflect"
	"testing"
)

func mustPanic(t *testing.T, want string, fn func()) {
	t.Helper()

	defer func() {
		r := recover()

		if r == nil {
			t.Fatalf("expected panic %q, but no panic occurred", want)
		}

		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected panic string, got %T", r)
		}

		if msg != want {
			t.Fatalf("expected panic %q, got %q", want, msg)
		}
	}()

	fn()
}

func TestMakeStrides(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  []int
	}{
		{
			name:  "0D shape",
			shape: []int{},
			want:  []int{},
		},
		{
			name:  "1D shape",
			shape: []int{5},
			want:  []int{1},
		},
		{
			name:  "2D shape",
			shape: []int{2, 3},
			want:  []int{3, 1},
		},
		{
			name:  "3D shape",
			shape: []int{2, 3, 4},
			want:  []int{12, 4, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strides := makeStrides(tt.shape...)

			if !reflect.DeepEqual(strides, tt.want) {
				t.Errorf("strides: want %v, got %v", tt.want, strides)
			}
		})
	}
}

func TestNew(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		wantPanic   string
		wantSize    int
		wantShape   []int
		wantStrides []int
	}{
		{
			name:      "invalid dimension",
			wantPanic: "tensor must have at least one dimension",
		},
		{
			name:      "invalid shape",
			shape:     []int{2, 0},
			wantPanic: "invalid shape",
		},
		{
			name:        "1D tensor",
			shape:       []int{5},
			wantSize:    5,
			wantShape:   []int{5},
			wantStrides: []int{1},
		},
		{
			name:        "2D tensor",
			shape:       []int{3, 4},
			wantSize:    12,
			wantShape:   []int{3, 4},
			wantStrides: []int{4, 1},
		},
		{
			name:        "3D tensor",
			shape:       []int{2, 5, 3},
			wantSize:    30,
			wantShape:   []int{2, 5, 3},
			wantStrides: []int{15, 3, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantPanic != "" {
				mustPanic(t, tt.wantPanic, func() { New(tt.shape...) })
				return
			}

			got := New(tt.shape...)

			if !reflect.DeepEqual(len(got.data), tt.wantSize) {
				t.Errorf("data length: want %v, got %v", tt.wantSize, got.data)
			}

			if !reflect.DeepEqual(got.shape, tt.wantShape) {
				t.Errorf("shape: want %v, got %v", tt.wantShape, got.shape)
			}

			if !reflect.DeepEqual(got.strides, tt.wantStrides) {
				t.Errorf("strides: want %v, got %v", tt.wantStrides, got.strides)
			}
		})
	}
}

func TestShape(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  []int
	}{
		{
			name:  "1D tensor",
			shape: []int{5},
			want:  []int{5},
		},
		{
			name:  "2D tensor",
			shape: []int{3, 4},
			want:  []int{3, 4},
		},
		{
			name:  "3D tensor",
			shape: []int{2, 5, 3},
			want:  []int{2, 5, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := New(tt.shape...)
			shape := got.Shape()

			if !reflect.DeepEqual(shape, tt.want) {
				t.Errorf("shape: want %v, got %v", tt.want, shape)
			}
		})
	}
}

func TestRank(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  int
	}{
		{
			name:  "1D tensor",
			shape: []int{5},
			want:  1,
		},
		{
			name:  "2D tensor",
			shape: []int{3, 4},
			want:  2,
		},
		{
			name:  "3D tensor",
			shape: []int{2, 5, 3},
			want:  3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := New(tt.shape...)
			rank := got.Rank()

			if !reflect.DeepEqual(rank, tt.want) {
				t.Errorf("rank: want %v, got %v", tt.want, rank)
			}
		})
	}
}

func TestSize(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  int
	}{
		{
			name:  "1D tensor",
			shape: []int{5},
			want:  5,
		},
		{
			name:  "2D tensor",
			shape: []int{3, 4},
			want:  12,
		},
		{
			name:  "3D tensor",
			shape: []int{2, 5, 3},
			want:  30,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := New(tt.shape...)
			size := got.Size()

			if !reflect.DeepEqual(size, tt.want) {
				t.Errorf("size: want %v, got %v", tt.want, size)
			}
		})
	}
}

func TestOffsetOf(t *testing.T) {
	tests := []struct {
		name      string
		shape     []int
		indices   []int
		wantPanic string
		want      int
	}{
		{
			name:      "too few indices",
			shape:     []int{2, 3},
			indices:   []int{1},
			wantPanic: "invalid indices",
		},
		{
			name:      "too many indices",
			shape:     []int{2, 3},
			indices:   []int{1, 1, 1},
			wantPanic: "invalid indices",
		},
		{
			name:      "negative index",
			shape:     []int{2, 3},
			indices:   []int{1, -1},
			wantPanic: "invalid indices",
		},
		{
			name:      "index equal dimension",
			shape:     []int{2, 3},
			indices:   []int{1, 3},
			wantPanic: "invalid indices",
		},
		{
			name:      "index out of bound",
			shape:     []int{2, 3},
			indices:   []int{1, 5},
			wantPanic: "invalid indices",
		},
		{
			name:    "1D first element",
			shape:   []int{5},
			indices: []int{0},
			want:    0,
		},
		{
			name:    "1D middle element",
			shape:   []int{5},
			indices: []int{3},
			want:    3,
		},
		{
			name:    "1D last element",
			shape:   []int{5},
			indices: []int{4},
			want:    4,
		},
		{
			name:    "2D first element",
			shape:   []int{2, 3},
			indices: []int{0, 0},
			want:    0,
		},
		{
			name:    "2D middle element",
			shape:   []int{2, 3},
			indices: []int{1, 1},
			want:    4,
		},
		{
			name:    "2D last element",
			shape:   []int{2, 3},
			indices: []int{1, 2},
			want:    5,
		},
		{
			name:    "3D first element",
			shape:   []int{2, 5, 3},
			indices: []int{0, 0, 0},
			want:    0,
		},
		{
			name:    "3D middle element",
			shape:   []int{2, 5, 3},
			indices: []int{1, 2, 2},
			want:    23,
		},
		{
			name:    "3D last element",
			shape:   []int{2, 5, 3},
			indices: []int{1, 4, 2},
			want:    29,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := New(tt.shape...)

			if tt.wantPanic != "" {
				mustPanic(t, tt.wantPanic, func() { tensor.offsetOf(tt.indices...) })
				return
			}

			offset := tensor.offsetOf(tt.indices...)

			if offset != tt.want {
				t.Errorf("offset: want %v, got %v", tt.want, offset)
			}
		})
	}
}

func TestAt(t *testing.T) {
	tests := []struct {
		name      string
		shape     []int
		indices   []int
		wantPanic string
		want      float64
	}{
		{
			name:      "too few indices",
			shape:     []int{2, 3},
			indices:   []int{1},
			wantPanic: "invalid indices",
		},
		{
			name:      "too many indices",
			shape:     []int{2, 3},
			indices:   []int{1, 1, 1},
			wantPanic: "invalid indices",
		},
		{
			name:      "negative index",
			shape:     []int{2, 3},
			indices:   []int{1, -1},
			wantPanic: "invalid indices",
		},
		{
			name:      "index equal dimension",
			shape:     []int{2, 3},
			indices:   []int{1, 3},
			wantPanic: "invalid indices",
		},
		{
			name:      "index out of bound",
			shape:     []int{2, 3},
			indices:   []int{1, 5},
			wantPanic: "invalid indices",
		},
		{
			name:    "1D first element",
			shape:   []int{5},
			indices: []int{0},
			want:    0.5,
		},
		{
			name:    "1D middle element",
			shape:   []int{5},
			indices: []int{3},
			want:    3.5,
		},
		{
			name:    "1D last element",
			shape:   []int{5},
			indices: []int{4},
			want:    4.5,
		},
		{
			name:    "2D first element",
			shape:   []int{2, 3},
			indices: []int{0, 0},
			want:    0.5,
		},
		{
			name:    "2D middle element",
			shape:   []int{2, 3},
			indices: []int{1, 1},
			want:    4.5,
		},
		{
			name:    "2D last element",
			shape:   []int{2, 3},
			indices: []int{1, 2},
			want:    5.5,
		},
		{
			name:    "3D first element",
			shape:   []int{2, 5, 3},
			indices: []int{0, 0, 0},
			want:    0.5,
		},
		{
			name:    "3D middle element",
			shape:   []int{2, 5, 3},
			indices: []int{1, 2, 2},
			want:    23.5,
		},
		{
			name:    "3D last element",
			shape:   []int{2, 5, 3},
			indices: []int{1, 4, 2},
			want:    29.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := New(tt.shape...)

			if tt.wantPanic != "" {
				mustPanic(t, tt.wantPanic, func() { tensor.At(tt.indices...) })
				return
			}

			for i := range tensor.data {
				tensor.data[i] = float64(i) + 0.5
			}

			value := tensor.At(tt.indices...)

			if value != tt.want {
				t.Errorf("value: want %v, got %v", tt.want, value)
			}
		})
	}
}

func TestSet(t *testing.T) {
	tests := []struct {
		name      string
		shape     []int
		indices   []int
		value     float64
		wantPanic string
		want      float64
	}{
		{
			name:      "too few indices",
			shape:     []int{2, 3},
			indices:   []int{1},
			wantPanic: "invalid indices",
		},
		{
			name:      "too many indices",
			shape:     []int{2, 3},
			indices:   []int{1, 1, 1},
			wantPanic: "invalid indices",
		},
		{
			name:      "negative index",
			shape:     []int{2, 3},
			indices:   []int{1, -1},
			wantPanic: "invalid indices",
		},
		{
			name:      "index equal dimension",
			shape:     []int{2, 3},
			indices:   []int{1, 3},
			wantPanic: "invalid indices",
		},
		{
			name:      "index out of bound",
			shape:     []int{2, 3},
			indices:   []int{1, 5},
			wantPanic: "invalid indices",
		},
		{
			name:    "1D first element",
			shape:   []int{5},
			indices: []int{0},
			value:   0.5,
			want:    0.5,
		},
		{
			name:    "1D middle element",
			shape:   []int{5},
			indices: []int{3},
			value:   3.5,
			want:    3.5,
		},
		{
			name:    "1D last element",
			shape:   []int{5},
			indices: []int{4},
			value:   4.5,
			want:    4.5,
		},
		{
			name:    "2D first element",
			shape:   []int{2, 3},
			indices: []int{0, 0},
			value:   0.5,
			want:    0.5,
		},
		{
			name:    "2D middle element",
			shape:   []int{2, 3},
			indices: []int{1, 1},
			value:   4.5,
			want:    4.5,
		},
		{
			name:    "2D last element",
			shape:   []int{2, 3},
			indices: []int{1, 2},
			value:   5.5,
			want:    5.5,
		},
		{
			name:    "3D first element",
			shape:   []int{2, 5, 3},
			indices: []int{0, 0, 0},
			value:   0.5,
			want:    0.5,
		},
		{
			name:    "3D middle element",
			shape:   []int{2, 5, 3},
			indices: []int{1, 2, 2},
			value:   23.5,
			want:    23.5,
		},
		{
			name:    "3D last element",
			shape:   []int{2, 5, 3},
			indices: []int{1, 4, 2},
			value:   29.5,
			want:    29.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := New(tt.shape...)

			if tt.wantPanic != "" {
				mustPanic(t, tt.wantPanic, func() { tensor.At(tt.indices...) })
				return
			}

			tensor.Set(tt.value, tt.indices...)

			value := tensor.At(tt.indices...)

			if value != tt.want {
				t.Errorf("value: want %v, got %v", tt.want, value)
			}
		})
	}
}

func TestReshape(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		newShape  []int
		wantPanic string
		want      []int
	}{
		{
			name:      "invalid dimension",
			tensor:    New(2, 3),
			newShape:  []int{},
			wantPanic: "tensor must have at least one dimension",
		},
		{
			name:      "tensor is non-contiguous",
			tensor:    New(2, 3).Transpose(),
			newShape:  []int{3, 2},
			wantPanic: "cannot reshape non-contiguous tensor",
		},
		{
			name:      "invalid dimension",
			tensor:    New(2, 3),
			newShape:  []int{0, 2},
			wantPanic: "invalid shape",
		},
		{
			name:      "reshape size mismatch",
			tensor:    New(2, 3),
			newShape:  []int{4, 5},
			wantPanic: "reshape size mismatch",
		},
		{
			name:     "same shape",
			tensor:   New(2, 3),
			newShape: []int{2, 3},
			want:     []int{2, 3},
		},
		{
			name:     "compatible shape",
			tensor:   New(4, 3),
			newShape: []int{2, 6},
			want:     []int{2, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantPanic != "" {
				mustPanic(t, tt.wantPanic, func() { _ = tt.tensor.Reshape(tt.newShape...) })
				return
			}

			newTensor := tt.tensor.Reshape(tt.newShape...)

			if !reflect.DeepEqual(newTensor.shape, tt.want) {
				t.Errorf("shape: want: %v, got: %v", tt.want, newTensor.shape)
			}
		})
	}
}

func TestTranspose(t *testing.T) {
	tensor := New(2, 3)

	tensor.Set(1, 0, 0)
	tensor.Set(2, 0, 1)
	tensor.Set(3, 0, 2)
	tensor.Set(4, 1, 0)
	tensor.Set(5, 1, 1)
	tensor.Set(6, 1, 2)

	got := tensor.Transpose()

	wantShape := []int{3, 2}
	if !reflect.DeepEqual(got.Shape(), wantShape) {
		t.Fatalf("shape: want %v, got %v", wantShape, got.Shape())
	}

	wantStrides := []int{1, 3}
	if !reflect.DeepEqual(got.strides, wantStrides) {
		t.Fatalf("strides: want %v, got %v", wantStrides, got.strides)
	}
}

func TestIsContiguous(t *testing.T) {
	tests := []struct {
		name   string
		tensor *Tensor
		want   bool
	}{
		{
			name:   "tensor is contiguous",
			tensor: New(2, 3, 4),
			want:   true,
		},
		{
			name:   "tensor is not contiguous",
			tensor: New(2, 3, 4).Transpose(),
			want:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.tensor.IsContiguous(); got != tt.want {
				t.Errorf("isContiguous: want: %v, got: %v", tt.want, got)
			}
		})
	}
}
