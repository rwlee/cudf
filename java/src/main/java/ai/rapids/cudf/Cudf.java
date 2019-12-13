/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.cudf;

/**
 * This is the binding class for cudf lib.
 */
class Cudf {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /* arith */

  static long binaryOp(ColumnVector lhs, ColumnVector rhs, BinaryOp op, DType outputType) {
    return binaryOpVV(lhs.getNativeCudfColumnAddress(), rhs.getNativeCudfColumnAddress(),
        op.nativeId, outputType.nativeId);
  }

  private static native long binaryOpVV(long lhs, long rhs, int op, int dtype);

  static long binaryOp(Scalar lhs, ColumnVector rhs, BinaryOp op, DType outputType) {
    return binaryOpSV(lhs.getScalarHandle(), rhs.getNativeCudfColumnAddress(),
        op.nativeId, outputType.nativeId);
  }

  private static native long binaryOpSV(long lhs, long rhs, int op, int dtype);

  static long binaryOp(ColumnVector lhs, Scalar rhs, BinaryOp op, DType outputType) {
    return binaryOpVS(lhs.getNativeCudfColumnAddress(), rhs.getScalarHandle(),
        op.nativeId, outputType.nativeId);
  }

  private static native long binaryOpVS(long lhs, long rhs, int op, int dtype);

  static Scalar reduce(ColumnVector v, ReductionOp op, DType outType) {
    return reduce(v.getNativeCudfColumnAddress(), op.nativeId, outType.nativeId);
  }

  private static native Scalar reduce(long v, int op, int dtype);
}
