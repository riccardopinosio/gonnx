package opset13

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var operators13 = map[string]func() ops.Operator{
	"Abs":             newAbs,
	"Acos":            newAcos,
	"Acosh":           newAcosh,
	"Add":             newAdd,
	"And":             newAnd,
	"ArgMax":          newArgMax,
	"Asin":            newAsin,
	"Asinh":           newAsinh,
	"Atan":            newAtan,
	"Atanh":           newAtanh,
	"Cast":            newCast,
	"Concat":          newConcat,
	"Constant":        newConstant,
	"ConstantOfShape": newConstantOfShape,
	"Conv":            newConv,
	"Cos":             newCos,
	"Cosh":            newCosh,
	"Div":             newDiv,
	"Equal":           newEqual,
	"Expand":          newExpand,
	"Flatten":         newFlatten,
	"Gather":          newGather,
	"Gemm":            newGemm,
	"Greater":         newGreater,
	"GreaterOrEqual":  newGreaterOrEqual,
	"GRU":             newGRU,
	"Less":            newLess,
	"LessOrEqual":     newLessOrEqual,
	"LinearRegressor": newLinearRegressor,
	"LogSoftmax":      newLogSoftmax,
	"LSTM":            newLSTM,
	"MatMul":          newMatMul,
	"Mul":             newMul,
	"Not":             newNot,
	"Or":              newOr,
	"PRelu":           newPRelu,
	"ReduceMax":       newReduceMax,
	"ReduceMin":       newReduceMin,
	"ReduceMean":      newReduceMean,
	"Relu":            newRelu,
	"Reshape":         newReshape,
	"RNN":             newRNN,
	"Scaler":          newScaler,
	"Shape":           newShape,
	"Sigmoid":         newSigmoid,
	"Sin":             newSin,
	"Sinh":            newSinh,
	"Slice":           newSlice,
	"Softmax":         newSoftmax,
	"Squeeze":         newSqueeze,
	"Sub":             newSub,
	"Tan":             newTan,
	"Tanh":            newTanh,
	"Transpose":       newTranspose,
	"Unsqueeze":       newUnsqueeze,
	"Xor":             newXor,
}

// GetOperator maps strings as found in the ModelProto to Operators from opset 13.
func GetOperator(operatorType string) (ops.Operator, error) {
	if opInit, ok := operators13[operatorType]; ok {
		return opInit(), nil
	}

	return nil, ops.ErrUnknownOperatorType(operatorType)
}

// GetOpNames returns a list with operator names for opset 13.
func GetOpNames() []string {
	opList := make([]string, 0, len(operators13))

	for opName := range operators13 {
		opList = append(opList, opName)
	}

	return opList
}
