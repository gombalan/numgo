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
