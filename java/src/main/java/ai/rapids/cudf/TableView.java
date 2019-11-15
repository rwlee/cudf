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

import java.util.ArrayList;
import java.util.List;

public class TableView implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final long rows;
  private long nativeHandle;
  private CudfColumnVector[] columns;

  /**
   * TableView class makes a copy of the array of {@link CudfColumnVector}s passed to it. The class
   * will decrease the refcount
   * on itself and all its contents when closed and free resources if refcount is zero
   * @param columns - Array of CudfColumnVectors
   */
  public TableView(CudfColumnVector... columns) {
    assert columns != null : "CudfColumnVectors can't be null";
    rows = columns.length > 0 ? columns[0].getRowCount() : 0;

    for (CudfColumnVector CudfColumnVector : columns) {
      assert (null != CudfColumnVector) : "CudfColumnVectors can't be null";
      assert (rows == CudfColumnVector.getRowCount()) : "All columns should have the same number of " +
          "rows " + CudfColumnVector.getType();
    }

    // Since Arrays are mutable objects make a copy
    this.columns = new CudfColumnVector[columns.length];
    long[] cudfColumnPointers = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      this.columns[i] = columns[i];
      columns[i].incRefCount();
      cudfColumnPointers[i] = columns[i].getNativeCudfColumnAddress();
    }

    nativeHandle = createCudfTableView(cudfColumnPointers);
  }

  public static final class OrderByArg {
    final int index;
    final boolean isDescending;
    final boolean isNullSmallest;

    OrderByArg(int index, boolean isDescending, boolean isNullSmallest) {
      this.index = index;
      this.isDescending = isDescending;
      this.isNullSmallest = isNullSmallest;
    }
  }

  public static TableView.OrderByArg asc(final int index) {
    return new TableView.OrderByArg(index, false, false);
  }

  public static TableView.OrderByArg desc(final int index) {
    return new TableView.OrderByArg(index, true, false);
  }

  public static TableView.OrderByArg asc(final int index, final boolean isNullSmallest) {
    return new TableView.OrderByArg(index, false, isNullSmallest);
  }

  public static TableView.OrderByArg desc(final int index, final boolean isNullSmallest) {
    return new TableView.OrderByArg(index, true, isNullSmallest);
  }

  public CudfColumnVector sortedOrder(TableView.OrderByArg... args) {
    assert args.length == columns.length || args.length == 0;
    boolean[] isDescending = new boolean[args.length];
    boolean[] isNullSmallest = new boolean[args.length];
    for (int i = 0; i < args.length; i++) {
      int index = args[i].index;
      assert (index >= 0 && index < columns.length) :
          "index is out of range 0 <= " + index + " < " + columns.length;
      isDescending[i] = args[i].isDescending;
      isNullSmallest[i] = args[i].isNullSmallest;
    }

    return new CudfColumnVector(sortedOrder(this.nativeHandle, isDescending, isNullSmallest));
  }

  public CudfColumnVector sortedOrder(boolean areNullSmallest, TableView.OrderByArg... args) {
    assert args.length == columns.length || args.length == 0;
    boolean[] isDescending = new boolean[args.length];
    boolean[] isNullSmallest = new boolean[args.length];
    for (int i = 0; i < args.length; i++) {
      int index = args[i].index;
      assert (index >= 0 && index < columns.length) :
          "index is out of range 0 <= " + index + " < " + columns.length;
      isDescending[i] = args[i].isDescending;
      isNullSmallest[i] = areNullSmallest;
    }

    return new CudfColumnVector(sortedOrder(this.nativeHandle, isDescending, isNullSmallest));
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      deleteCudfTableView(nativeHandle);
      nativeHandle = 0;
    }
    if (columns != null) {
      for (int i = 0; i < columns.length; i++) {
        columns[i].close();
        columns[i] = null;
      }
      columns = null;
    }
  }


  ////////////////////////////////////////////////////////////////////////////
  // NATIVE
  ////////////////////////////////////////////////////////////////////////////

  private native long createCudfTableView(long[] nativeColumnHandles);
  private native long deleteCudfTableView(long nativeHandle);
  private static native long sortedOrder(long input, boolean[] isDescending, boolean[] areNullsSmallest);


  /////////////////////////////////////////////////////////////////////////////
  // BUILDER
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Create a table on the GPU with data from the CPU.  This is not fast and intended mostly for
   * tests.
   */
  public static final class TestBuilder {
    private final List<TypeId> types = new ArrayList<>();
    private final List<Object> typeErasedData = new ArrayList<>();

    public TableView.TestBuilder column(String... values) {
      types.add(TypeId.STRING);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder column(Boolean... values) {
      types.add(TypeId.BOOL8);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder column(Byte... values) {
      types.add(TypeId.INT8);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder column(Short... values) {
      types.add(TypeId.INT16);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder column(Integer... values) {
      types.add(TypeId.INT32);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder column(Long... values) {
      types.add(TypeId.INT64);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder column(Float... values) {
      types.add(TypeId.FLOAT32);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder column(Double... values) {
      types.add(TypeId.FLOAT64);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder timestampDayColumn(Long... values) {
      types.add(TypeId.TIMESTAMP_DAYS);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder timestampNanosecondsColumn(Long... values) {
      types.add(TypeId.TIMESTAMP_NANOSECONDS);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder timestampMillisecondsColumn(Long... values) {
      types.add(TypeId.TIMESTAMP_MILLISECONDS);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder timestampMicrosecondsColumn(Long... values) {
      types.add(TypeId.TIMESTAMP_MICROSECONDS);
      typeErasedData.add(values);
      return this;
    }

    public TableView.TestBuilder timestampSecondsColumn(Long... values) {
      types.add(TypeId.TIMESTAMP_SECONDS);
      typeErasedData.add(values);
      return this;
    }

    private static CudfColumnVector from(TypeId type, Object dataArray) {
      CudfColumnVector ret = null;
      switch (type) {
        case STRING:
          ret = CudfColumnVector.fromStrings((String[]) dataArray);
          break;
        case BOOL8:
          throw new UnsupportedOperationException("op not supported");
//          ret = CudfColumnVector.fromBoxedBooleans((Boolean[]) dataArray);
//          break;
        case INT8:
          throw new UnsupportedOperationException("op not supported");
//          ret = CudfColumnVector.fromBoxedBytes((Byte[]) dataArray);
//          break;
        case INT16:
          throw new UnsupportedOperationException("op not supported");
//          ret = CudfColumnVector.fromBoxedShorts((Short[]) dataArray);
//          break;
        case INT32:
          ret = CudfColumnVector.fromBoxedInts((Integer[]) dataArray);
          break;
        case INT64:
          throw new UnsupportedOperationException("op not supported");
//          ret = CudfColumnVector.fromBoxedLongs((Long[]) dataArray);
//          break;
        case TIMESTAMP_DAYS:
          throw new UnsupportedOperationException("op not supported");
//          ret = CudfColumnVector.timestampsFromBoxedLongs(unit, (Long[]) dataArray);
//          break;
        case FLOAT32:
          throw new UnsupportedOperationException("op not supported");
//          ret = CudfColumnVector.fromBoxedFloats((Float[]) dataArray);
//          break;
        case FLOAT64:
          throw new UnsupportedOperationException("op not supported");
//          ret = CudfColumnVector.fromBoxedDoubles((Double[]) dataArray);
//          break;
        default:
          throw new IllegalArgumentException(type + " is not supported yet");
      }
      return ret;
    }

    public TableView build() {
      List<CudfColumnVector> columns = new ArrayList<>(types.size());
      try {
        for (int i = 0; i < types.size(); i++) {
          columns.add(from(types.get(i), typeErasedData.get(i)));
        }
        for (CudfColumnVector cv : columns) {
          cv.ensureOnDevice();
        }
        return new TableView(columns.toArray(new CudfColumnVector[columns.size()]));
      } finally {
        for (CudfColumnVector cv : columns) {
          cv.close();
        }
      }
    }
  }
}
