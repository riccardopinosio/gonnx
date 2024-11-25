package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

const (
	MinReduceMeanAttributes = 1
	MaxReduceMeanAttributes = 2
)

// ReduceMean represents the ONNX reduceMean operator.
type ReduceMean struct {
	axes     []int
	keepDims bool
}

// newReduceMean creates a new reduceMean operator.
func newReduceMean() ops.Operator {
	return &ReduceMean{
		axes:     []int{},
		keepDims: true,
	}
}

// Init initializes the reduceMin operator.
func (r *ReduceMean) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	if len(attributes) == 0 || len(attributes) > MaxReduceMeanAttributes {
		return ops.ErrInvalidOptionalAttributeCount(MinReduceMeanAttributes, MaxReduceMeanAttributes, len(attributes), r)
	}

	for _, attr := range attributes {
		switch attr.GetName() {
		case "axes":
			axes, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}

			r.axes = axes
		case "keepdims":
			r.keepDims = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), r)
		}
	}

	return nil
}

func cast(count int, t tensor.Dtype) (any, error) {
	switch t {
	case tensor.Int:
		return count, nil
	case tensor.Int8:
		return int8(count), nil
	case tensor.Int16:
		return int16(count), nil
	case tensor.Int32:
		return int32(count), nil
	case tensor.Int64:
		return int64(count), nil
	case tensor.Uint:
		return uint(count), nil
	case tensor.Uint8:
		return uint8(count), nil
	case tensor.Uint16:
		return uint16(count), nil
	case tensor.Uint32:
		return uint32(count), nil
	case tensor.Uint64:
		return uint64(count), nil
	case tensor.Float32:
		return float32(count), nil
	case tensor.Float64:
		return float64(count), nil
	case tensor.Complex64:
		return complex(float32(count), 0), nil
	case tensor.Complex128:
		return complex(float64(count), 0), nil
	default:
		return nil, errors.Errorf("No methods found for Sum for %v", t)
	}
}

// Apply applies the reduceMean operator.
func (r *ReduceMean) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := tensor.New(tensor.WithBacking(inputs[0].Data()), tensor.WithShape(inputs[0].Shape()...))

	axes := make([]int, len(r.axes))
	for i, axis := range r.axes {
		axes[i] = ops.ConvertNegativeAxis(axis, len(input.Shape()))
	}

	sum, err := input.Sum(axes...)
	if err != nil {
		return nil, err
	}

	// tensor.NonMaskedCount seems to be bugged, so we'll calculate the count manually
	count := 1
	for _, axis := range axes {
		count *= input.Shape()[axis]
	}

	countCast, err := cast(count, sum.Dtype())
	if err != nil {
		return nil, err
	}

	out, err := sum.DivScalar(countCast, true)

	if err != nil {
		return nil, err
	}

	if r.keepDims {
		newShape := input.Shape()
		for _, axes := range axes {
			newShape[axes] = 1
		}

		err := out.Reshape(newShape...)
		if err != nil {
			return nil, err
		}
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (r *ReduceMean) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(r, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (r *ReduceMean) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (r *ReduceMean) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (r *ReduceMean) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint8, tensor.Int8, tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (r *ReduceMean) String() string {
	return "reduceMean operator"
}
