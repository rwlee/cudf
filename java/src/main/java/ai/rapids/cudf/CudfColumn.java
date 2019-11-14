/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

public class CudfColumn {
  public final long nativeHandle;

  public CudfColumn(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  public CudfColumn(TypeId typeId, int rows, MaskState maskState) {
    this.nativeHandle = makeNumericCudfColumn(typeId.nativeId, rows, maskState.nativeId);
  }

  public long getNativeValidPointer() {
    return getNativeValidPointer(nativeHandle);
  }

  public long getNativeRowCount() {
    return getNativeRowCount(nativeHandle);
  }

  public long getNativeNullCount() {
    return getNativeNullCount(nativeHandle);
  }

  public long getNativeDataPointer() {
    return getNativeDataPointer(nativeHandle);
  }

  public void deleteCudfColumn() {
    deleteCudfColumn(nativeHandle);
  }

  public long transform(String udf, boolean isPtx) {
    return transform(this.nativeHandle, udf, isPtx);
  }

  //////////////////////////////////////////////////////////////////////////////
  // NATIVE METHODS
  /////////////////////////////////////////////////////////////////////////////
  private native void deleteCudfColumn(long cudfColumnHandle) throws CudfException;

  static native int getNativeTypeId(long cudfColumnHandle) throws CudfException;

  private native int getNativeRowCount(long cudfColumnHandle) throws CudfException;

  private native int getNativeNullCount(long cudfColumnHandle) throws CudfException;

  private native long getNativeDataPointer(long cudfColumnHandle) throws CudfException;

  private native long getNativeValidPointer(long cudfColumnHandle) throws CudfException;

  private native long makeNumericCudfColumn(int type, int rows, int maskState);

  static native long getNativeValidPointerSize(int size);

  private native static long transform(long handle, String udf, boolean isPtx);

}
