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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.function.Consumer;

/**
 * A Column Vector. This class represents the immutable vector of data.  This class holds
 * references to off heap memory and is reference counted to know when to release it.  Call
 * close to decrement the reference count when you are done with the column, and call inRefCount
 * to increment the reference count.
 */
public final class CudfColumnVector implements AutoCloseable {
  private static final String STRING_NOT_SUPPORTED = "libCudf++ Strings are not supported in Java\n";
  /**
   * The size in bytes of an offset entry
   */
  static final int OFFSET_SIZE = TypeId.INT32.sizeInBytes;
  private static final Logger log = LoggerFactory.getLogger(CudfColumnVector.class);

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final TypeId type;
  private final OffHeapState offHeap = new OffHeapState();
  private long rows;
  private long nullCount;
  private int refCount;

  /**
   * Wrap an existing on device gdf_column with the corresponding CudfColumn.
   */
  CudfColumnVector(long nativePointer) {
    assert nativePointer != 0;
    MemoryCleaner.register(this, offHeap);
    offHeap.cudfColumnHandle = new CudfColumn(nativePointer);
    this.type = getTypeId(nativePointer);
    offHeap.setHostData(null);
    this.rows = offHeap.cudfColumnHandle.getNativeRowCount();
    this.nullCount = offHeap.cudfColumnHandle.getNativeNullCount();
    DeviceMemoryBufferView data = null;
    DeviceMemoryBufferView offsets = null;
    if (type != TypeId.STRING) {
      data = new DeviceMemoryBufferView(offHeap.cudfColumnHandle.getNativeDataPointer(), this.rows * type.sizeInBytes);
    } else {
      long[] dataAndOffsets = CudfColumn.getStringDataAndOffsets(getNativeCudfColumnAddress());
      data = new DeviceMemoryBufferView(dataAndOffsets[0], dataAndOffsets[1]);
      offsets = new DeviceMemoryBufferView(dataAndOffsets[2], dataAndOffsets[3]);
    }
    DeviceMemoryBufferView valid = null;
    long validPointer = offHeap.cudfColumnHandle.getNativeValidPointer();
    if (validPointer != 0) {
      valid = new DeviceMemoryBufferView(validPointer, CudfColumn.getNativeValidPointerSize((int) rows));
    }
    this.offHeap.setDeviceData(new BufferEncapsulator<>(data, valid, offsets));
    this.refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * Create a new column vector with data populated on the host.
   */
  CudfColumnVector(TypeId type, long rows, long nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer) {
    this(type, rows, nullCount, hostDataBuffer, hostValidityBuffer, null);
  }

  /**
   * Create a new column vector with data populated on the host.
   * @param type               the type of the vector
   * @param rows               the number of rows in the vector.
   * @param nullCount          the number of nulls in the vector.
   * @param hostDataBuffer     The host side data for the vector. In the case of STRING and
   *                           STRING_CATEGORY this is the string data stored as bytes.
   * @param hostValidityBuffer arrow like validity buffer 1 bit per row, with padding for
   *                           64-bit alignment.
   * @param offsetBuffer       only valid for STRING and STRING_CATEGORY this is the offsets into
   *                           the hostDataBuffer indicating the start and end of a string
   *                           entry. It should be (rows + 1) ints.
   */
  CudfColumnVector(TypeId type, long rows, long nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
                   HostMemoryBuffer offsetBuffer) {
    if (nullCount > 0 && hostValidityBuffer == null) {
      throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
    }
    MemoryCleaner.register(this, offHeap);
    offHeap.setHostData(new BufferEncapsulator(hostDataBuffer, hostValidityBuffer, offsetBuffer));
    offHeap.setDeviceData(null);
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;
    refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * Returns the type of this vector.
   */
  public TypeId getType() {
    return type;
  }

  private static TypeId getTypeId(long cudfColumnHandle) throws CudfException {
    return TypeId.fromNative(CudfColumn.getNativeTypeId(cudfColumnHandle));
  }

  /**
   * USE WITH CAUTION: This method exposes the address of the native cudf_column.  This allows
   * writing custom kernels or other cuda operations on the data.  DO NOT close this column
   * vector until you are completely done using the native column.  DO NOT modify the column in
   * any way.  This should be treated as a read only data structure. This API is unstable as
   * the underlying C/C++ API is still not stabilized.  If the underlying data structure
   * is renamed this API may be replaced.  The underlying data structure can change from release
   * to release (it is not stable yet) so be sure that your native code is complied against the
   * exact same version of libcudf as this is released for.
   */
  public final long getNativeCudfColumnAddress() {
    if (offHeap.cudfColumnHandle == null) {
      //Data must have been dropped from device recreate it on device
      assert rows <= Integer.MAX_VALUE;
      assert getNullCount() <= Integer.MAX_VALUE;
      ensureOnDevice();
    }
    return offHeap.cudfColumnHandle.nativeHandle;
  }

  /**
   * Increment the reference count for this column.  You need to call close on this
   * to decrement the reference count again.
   */
  public CudfColumnVector incRefCount() {
    return incRefCountInternal(false);
  }

  private CudfColumnVector incRefCountInternal(boolean isFirstTime) {
    offHeap.addRef();
    if (refCount <= 0 && !isFirstTime) {
      offHeap.logRefCountDebug("INC AFTER CLOSE " + this);
      throw new IllegalStateException("Column is already closed");
    }
    refCount++;
    return this;
  }

  /**
   * Returns the number of rows in this vector.
   */
  public long getRowCount() {
    return rows;
  }

  /**
   * Return true if the data is on the host, or false if it is not. Note that
   * if there are no rows there is no data to be on the host, but this will
   * still return true.
   */
  public boolean hasHostData() {
    return offHeap.getHostData() != null || rows == 0;
  }

  private void checkHasHostData() {
    if (!hasHostData()) {
      if (refCount <= 0) {
        throw new IllegalStateException("Vector was already closed.");
      }
      throw new IllegalStateException("Vector not on host");
    }
  }

  /**
   * Returns if the vector has nulls.
   */
  public boolean hasNulls() {
    return getNullCount() > 0;
  }

  /**
   * Returns the number of nulls in the data.
   */
  public long getNullCount() {
    return nullCount;
  }

  /**
   * Return true if the data is on the device, or false if it is not. Note that
   * if there are no rows there is no data to be on the device, but this will
   * still return true.
   */
  public boolean hasDeviceData() {
    return offHeap.getDeviceData() != null || rows == 0;
  }

  private void checkHasDeviceData() {
    if (!hasDeviceData()) {
      if (refCount <= 0) {
        throw new IllegalStateException("Vector was already closed.");
      }
      throw new IllegalStateException("Vector not on device");
    }
  }

  /**
   * Drop any data stored on the host, but move it to the device first if necessary.
   */
  public final void dropHostData() {
    ensureOnDevice();
    if (offHeap.hostData != null) {
      offHeap.hostData.close();
      offHeap.hostData = null;
    }
  }

  /**
   * Be sure the data is on the host.
   */
  public final void ensureOnHost() {
    if (offHeap.getHostData() == null && rows != 0) {
      checkHasDeviceData();

      try (NvtxRange toHost = new NvtxRange("ensureOnHost", NvtxColor.BLUE)) {
        HostMemoryBuffer hostDataBuffer = null;
        HostMemoryBuffer hostValidityBuffer = null;
        HostMemoryBuffer hostOffsetsBuffer = null;
        boolean needsCleanup = true;
        try {
          if (offHeap.getDeviceData().valid != null) {
            hostValidityBuffer = HostMemoryBuffer.allocate(offHeap.getDeviceData().valid.getLength());
          }
          if (type == TypeId.STRING) {
            hostOffsetsBuffer = HostMemoryBuffer.allocate(offHeap.deviceData.offsets.length);
          }
          hostDataBuffer = HostMemoryBuffer.allocate(offHeap.getDeviceData().data.getLength());

          offHeap.setHostData(new BufferEncapsulator(hostDataBuffer, hostValidityBuffer,hostOffsetsBuffer));
          needsCleanup = false;
        } finally {
          if (needsCleanup) {
            if (hostDataBuffer != null) {
              hostDataBuffer.close();
            }
            if (hostValidityBuffer != null) {
              hostValidityBuffer.close();
            }
          }
        }
        offHeap.getHostData().data.copyFromDeviceBuffer(offHeap.getDeviceData().data);
        if (offHeap.getHostData().valid != null) {
          offHeap.getHostData().valid.copyFromDeviceBuffer(offHeap.getDeviceData().valid);
        }
        if (type == TypeId.STRING) {
          offHeap.hostData.offsets.copyFromDeviceBuffer(offHeap.deviceData.offsets);
        }
      }
    }
  }

  /**
   * Be sure the data is on the device.
   */
  public final void ensureOnDevice() {
    if (offHeap.getDeviceData() == null && rows != 0) {
      checkHasHostData();

      assert type != TypeId.STRING || offHeap.getHostData().offsets != null;

      try (NvtxRange toDev = new NvtxRange("ensureOnDevice", NvtxColor.BLUE)) {
        DeviceMemoryBufferView deviceDataBuffer = null;
        DeviceMemoryBufferView deviceValidityBuffer = null;
        DeviceMemoryBufferView deviceOffsetsBuffer = null;

        boolean needsCleanup = true;
        try {
          if (type != TypeId.STRING) {
            offHeap.cudfColumnHandle = new CudfColumn(type, (int) rows, hasNulls() ? MaskState.UNINITIALIZED : MaskState.UNALLOCATED);
            deviceDataBuffer = new DeviceMemoryBufferView(offHeap.cudfColumnHandle.getNativeDataPointer(), rows * type.sizeInBytes);
          } else {
            offHeap.cudfColumnHandle = new CudfColumn(offHeap.hostData.data.address, offHeap.hostData.offsets.address, offHeap.hostData.valid == null ? 0 : offHeap.hostData.valid.address, (int) nullCount, (int) rows);
            long[] dataAndOffsets = CudfColumn.getStringDataAndOffsets(getNativeCudfColumnAddress());
            deviceDataBuffer = new DeviceMemoryBufferView(dataAndOffsets[0], dataAndOffsets[1]);
            deviceOffsetsBuffer = new DeviceMemoryBufferView(dataAndOffsets[2], dataAndOffsets[3]);
          }
          if (hasNulls()) {
            deviceValidityBuffer = new DeviceMemoryBufferView(offHeap.cudfColumnHandle.getNativeValidPointer(), offHeap.cudfColumnHandle.getNativeValidPointerSize((int) rows));
          }
          offHeap.setDeviceData(new BufferEncapsulator(deviceDataBuffer, deviceValidityBuffer, deviceOffsetsBuffer));
          needsCleanup = false;
        } finally {
          if (needsCleanup) {
            if (deviceDataBuffer != null) {
              deviceDataBuffer.close();
            }
            if (deviceValidityBuffer != null) {
              deviceValidityBuffer.close();
            }
          }
        }
        if (type != TypeId.STRING) {
          if (offHeap.getDeviceData().valid != null) {
            offHeap.getDeviceData().valid.copyFromHostBuffer(offHeap.getHostData().valid);
          }

          offHeap.getDeviceData().data.copyFromHostBuffer(offHeap.getHostData().data);
        }
      }
    }
  }

  private static CudfColumnVector fromStrings(TypeId type, String... values) {
    assert type == TypeId.STRING;
    int rows = values.length;
    long nullCount = 0;
    // How many bytes do we need to hold the data.  Sorry this is really expensive
    long bufferSize = 0;
    for (String s: values) {
      if (s == null) {
        nullCount++;
      } else {
        bufferSize += s.getBytes(StandardCharsets.UTF_8).length;
      }
    }
    if (nullCount > 0) {
      return build(type, rows, bufferSize, (b) -> b.appendBoxed(values));
    }
    return build(type, rows, bufferSize, (b) -> {
      for (String s: values) {
        b.append(s);
      }
    });
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls. This is really intended to be used only for testing as
   * it is slow and memory intensive to translate between java strings and UTF8 strings.
   */
  public static CudfColumnVector fromStrings(String... values) {
    return fromStrings(TypeId.STRING, values);
  }

  /**
   * Create a new vector from the given values.
   */
  public static CudfColumnVector fromInts(int... values) {
    return build(TypeId.INT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static CudfColumnVector fromBoxedInts(Integer... values) {
    return build(TypeId.INT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when
   * done with it. Please try to use {@see #build(int, Consumer)} instead to avoid needing to
   * close the builder.
   * @param type the type of vector to build.
   * @param rows the number of rows this builder can hold
   * @return the builder to use.
   */
  public static CudfColumnVector.Builder builder(TypeId type, int rows) {
    return new CudfColumnVector.Builder(type, rows, 0);
  }

  /**
   * Create a new Builder to hold the specified number of rows and with enough space to hold the
   * given amount of string data. Be sure to close the builder when done with it. Please try to
   * use {@see #build(int, int, Consumer)} instead to avoid needing to close the builder.
   * @param type the type of vector to build.
   * @param rows the number of rows this builder can hold
   * @param stringBufferSize the size of the string buffer to allocate.
   * @return the builder to use.
   */
  public static CudfColumnVector.Builder builder(TypeId type, int rows, long stringBufferSize) {
    assert type == TypeId.STRING;
    return new CudfColumnVector.Builder(type, rows, stringBufferSize);
  }

  /**
   * Create a new vector.
   * @param type       the type of vector to build.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static CudfColumnVector build(TypeId type, int rows,
                                   Consumer<CudfColumnVector.Builder> init) {
    try (CudfColumnVector.Builder builder = builder(type, rows)) {
      init.accept(builder);
      return builder.build();
    }
  }

  public static CudfColumnVector build(TypeId type, int rows, long stringBufferSize, Consumer<CudfColumnVector.Builder> init) {
    try (CudfColumnVector.Builder builder = builder(type, rows, stringBufferSize)) {
      init.accept(builder);
      return builder.build();
    }
  }

  /**
   * Check if the value at index is null or not.
   */
  public boolean isNull(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    if (hasNulls()) {
      checkHasHostData();
      return BitVectorHelper.isNull(offHeap.getHostData().valid, index);
    }
    return false;
  }

  /**
   * Generic type independent asserts when getting a value from a single index.
   * @param index where to get the data from.
   */
  private void assertsForGet(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert offHeap.getHostData() != null : "data is not on the host";
    assert !isNull(index) : " value at " + index + " is null";
  }

  /**
   * Get the value at index.
   */
  public final int getInt(long index) {
    assert type == TypeId.INT32 || type == TypeId.TIMESTAMP_DAYS;
    assertsForGet(index);
    return offHeap.getHostData().data.getInt(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.  This API is slow as it has to translate the
   * string representation.  Please use it with caution.
   */
  public String getJavaString(long index) {
    assert type == TypeId.STRING;
    assertsForGet(index);
    int start = offHeap.getHostData().offsets.getInt(index * OFFSET_SIZE);
    int size = offHeap.getHostData().offsets.getInt((index + 1) * OFFSET_SIZE) - start;
    byte[] rawData = new byte[size];
    if (size > 0) {
      offHeap.getHostData().data.getBytes(rawData, 0, start, size);
    }
    return new String(rawData, StandardCharsets.UTF_8);
  }

  public CudfColumnVector transform(String udf, boolean isPtx) {
    return new CudfColumnVector(offHeap.cudfColumnHandle.transform(udf, isPtx));
  }

  /**
   * Close this Vector and free memory allocated for HostMemoryBuffer and DeviceMemoryBufferView
   */
  @Override
  public final void close() {
    refCount--;
    offHeap.delRef();
    if (refCount == 0) {
      offHeap.clean(false);
    } else if (refCount < 0) {
      log.error("Close called too many times on {}", this);
      offHeap.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times");
    }
  }

  @Override
  public String toString() {
    return "CudfColumn{" +
        "rows=" + rows +
        ", type=" + type +
        ", hostData=" + offHeap.getHostData() +
        ", deviceData=" + offHeap.getDeviceData() +
        ", nullCount=" + nullCount +
        ", cudfColumn=" + offHeap.cudfColumnHandle.nativeHandle +
        '}';
  }

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Encapsulator class to hold the two buffers as a cohesive object
   */
  private static final class BufferEncapsulator<T extends MemoryBuffer> implements AutoCloseable {
    public final T data;
    public final T valid;
    public final T offsets;

    BufferEncapsulator(T data, T valid, T offsets) {
      this.data = data;
      this.valid = valid;
      this.offsets = offsets;
    }

    @Override
    public String toString() {
      T type = data == null ? valid : data;
      type = type == null ? offsets : type;
      String t = "UNKNOWN";
      if (type != null) {
        t = type.getClass().getSimpleName();
      }
      return "BufferEncapsulator{type= " + t
          + ", data= " + data
          + ", valid= " + valid
          + ", offsets= " + offsets + "}";
    }

    @Override
    public void close() {
      if (data != null) {
        data.close();
      }
      if (valid != null) {
        valid.close();
      }
      if (offsets != null) {
        offsets.close();
      }
    }

    /**
     * This is a really ugly API, but it is possible that the lifecycle of a column of
     * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
     * tracking code that this is expected for this column, and big scary warnings should
     * not be printed when this happens.
     */
    public void noWarnLeakExpected() {
      if (data != null) {
        data.noWarnLeakExpected();
      }
      if (valid != null) {
        valid.noWarnLeakExpected();
      }
      if (offsets != null) {
        offsets.noWarnLeakExpected();
      }
    }
  }

  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  protected static final class OffHeapState extends MemoryCleaner.Cleaner {
    private BufferEncapsulator<HostMemoryBuffer> hostData;
    private BufferEncapsulator<DeviceMemoryBufferView> deviceData;
    private CudfColumn cudfColumnHandle;

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (getHostData() != null) {
        getHostData().close();
        setHostData(null);
        neededCleanup = true;
      }
      if (getDeviceData() != null) {
        setDeviceData(null);
        neededCleanup = true;
      }
      if (cudfColumnHandle != null) {
        cudfColumnHandle.deleteCudfColumn();
        cudfColumnHandle = null;
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("YOU LEAKED A COLUMN VECTOR!!!!");
        logRefCountDebug("Leaked vector");
      }
      return neededCleanup;
    }

    public BufferEncapsulator<HostMemoryBuffer> getHostData() {
      return hostData;
    }

    public void setHostData(BufferEncapsulator<HostMemoryBuffer> hostData) {
      if (isLeakExpected() && hostData != null) {
        hostData.noWarnLeakExpected();
      }
      this.hostData = hostData;
    }

    public BufferEncapsulator<DeviceMemoryBufferView> getDeviceData() {
      return deviceData;
    }

    /**
     * This returns total memory allocated in device for the CudfColumn.
     * NOTE: If TypeId is STRING_CATEGORY, the size is estimated. The estimate assumes the length
     * of strings to be 10 characters in each row and returns 24 bytes per dictionary entry.
     * @param type
     * @return number of device bytes allocated for this column
     */
    public long getDeviceMemoryLength(TypeId type) {
      long length = 0;
      if (deviceData != null) {
        length = deviceData.valid != null ? deviceData.valid.getLength() : 0;
        if (type == TypeId.STRING) {
          throw new UnsupportedOperationException(STRING_NOT_SUPPORTED);
//          length += getDeviceMemoryStringSize(cudfColumnHandle.nativeHandle);
        } else {
          length += deviceData.data != null ? deviceData.data.getLength() : 0;
        }
      }
      return length;
    }

    public void setDeviceData(BufferEncapsulator<DeviceMemoryBufferView> deviceData) {
      if (isLeakExpected() && deviceData != null) {
        deviceData.noWarnLeakExpected();
      }
      this.deviceData = deviceData;
    }
  }

  /**
   * Build
   */
  public static final class Builder implements AutoCloseable {
    private final long rows;
    private final TypeId type;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private HostMemoryBuffer offsets;
    private long currentIndex = 0;
    private long nullCount;
    private long stringBufferSize = 0;
    private int currentStringByteIndex = 0;
    private boolean built;

    /**
     * Create a builder with a buffer of size rows
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param stringBufferSize the size of the string data buffer if we are
     *                         working with Strings.  It is ignored otherwise.
     */
    Builder(TypeId type, long rows, long stringBufferSize) {
      this.type = type;
      this.rows = rows;
      if (type == TypeId.STRING) {
        if (stringBufferSize <= 0) {
          // We need at least one byte or we will get NULL back for data
          stringBufferSize = 1;
        }
        this.data = HostMemoryBuffer.allocate(stringBufferSize);
        // The offsets are ints and there is 1 more than the number of rows.
        this.offsets = HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE);
        // The first offset is always 0
        this.offsets.setInt(0, 0);
        this.stringBufferSize = stringBufferSize;
      } else {
        this.data = HostMemoryBuffer.allocate(rows * type.sizeInBytes);
      }
    }

    /**
     * Create a builder with a buffer of size rows (for testing ONLY).
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param testData   a buffer to hold the data (should be large enough to hold rows entries).
     * @param testValid  a buffer to hold the validity vector (should be large enough to hold
     *                   rows entries or is null).
     * @param testOffsets a buffer to hold the offsets for strings and string categories.
     */
    Builder(TypeId type, long rows, HostMemoryBuffer testData,
            HostMemoryBuffer testValid, HostMemoryBuffer testOffsets) {
      this.type = type;
      this.rows = rows;
      this.data = testData;
      this.valid = testValid;
    }

    public final Builder append(boolean value) {
      assert type == TypeId.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value ? (byte)1 : (byte)0);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value) {
      assert type == TypeId.INT8 || type == TypeId.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value, long count) {
      assert (count + currentIndex) <= rows;
      assert type == TypeId.INT8 || type == TypeId.BOOL8;
      data.setMemory(currentIndex * type.sizeInBytes, count, value);
      currentIndex += count;
      return this;
    }

    public final Builder append(short value) {
      assert type == TypeId.INT16;
      assert currentIndex < rows;
      data.setShort(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(int value) {
      assert (type == TypeId.INT32 || type == TypeId.TIMESTAMP_DAYS);
      assert currentIndex < rows;
      data.setInt(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(long value) {
      assert type == TypeId.INT64 || type == TypeId.TIMESTAMP_MILLISECONDS ||
          type == TypeId.TIMESTAMP_MICROSECONDS || type == TypeId.TIMESTAMP_NANOSECONDS ||
          type == TypeId.TIMESTAMP_SECONDS;
      assert currentIndex < rows;
      data.setLong(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(float value) {
      assert type == TypeId.FLOAT32;
      assert currentIndex < rows;
      data.setFloat(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(double value) {
      assert type == TypeId.FLOAT64;
      assert currentIndex < rows;
      data.setDouble(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public Builder append(String value) {
      assert value != null : "appendNull must be used to append null strings";
      return appendUTF8String(value.getBytes(StandardCharsets.UTF_8));
    }

    public Builder appendUTF8String(byte[] value) {
      return appendUTF8String(value, 0, value.length);
    }

    public Builder appendUTF8String(byte[] value, int offset, int length) {
      assert value != null : "appendNull must be used to append null strings";
      assert offset >= 0;
      assert length >= 0;
      assert value.length + offset <= length;
      assert type == TypeId.STRING;
      assert currentIndex < rows;
      // just for strings we want to throw a real exception if we would overrun the buffer
      long oldLen = data.getLength();
      long newLen = oldLen;
      while (currentStringByteIndex + length > newLen) {
        newLen *= 2;
      }
      if (newLen > Integer.MAX_VALUE) {
        throw new IllegalStateException("A string buffer is not supported over 2GB in size");
      }
      if (newLen != oldLen) {
        // need to grow the size of the buffer.
        HostMemoryBuffer newData = HostMemoryBuffer.allocate(newLen);
        try {
          newData.copyFromHostBuffer(0, data, 0, currentStringByteIndex);
          data.close();
          data = newData;
          newData = null;
        } finally {
          if (newData != null) {
            newData.close();
          }
        }
      }
      if (length > 0) {
        data.setBytes(currentStringByteIndex, value, offset, length);
      }
      currentStringByteIndex += length;
      currentIndex++;
      offsets.setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      return this;
    }

    public Builder appendArray(byte... values) {
      assert (values.length + currentIndex) <= rows;
      assert type == TypeId.INT8 || type == TypeId.BOOL8;
      data.setBytes(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(short... values) {
      assert type == TypeId.INT16;
      assert (values.length + currentIndex) <= rows;
      data.setShorts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(int... values) {
      assert (type == TypeId.INT32 || type == TypeId.TIMESTAMP_DAYS);
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(long... values) {
      assert type == TypeId.INT64 || type == TypeId.TIMESTAMP_MILLISECONDS ||
          type == TypeId.TIMESTAMP_MICROSECONDS || type == TypeId.TIMESTAMP_NANOSECONDS ||
          type == TypeId.TIMESTAMP_SECONDS;
      assert (values.length + currentIndex) <= rows;
      data.setLongs(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(float... values) {
      assert type == TypeId.FLOAT32;
      assert (values.length + currentIndex) <= rows;
      data.setFloats(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(double... values) {
      assert type == TypeId.FLOAT64;
      assert (values.length + currentIndex) <= rows;
      data.setDoubles(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Byte... values) throws IndexOutOfBoundsException {
      for (Byte b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Boolean... values) throws IndexOutOfBoundsException {
      for (Boolean b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b ? (byte) 1 : (byte) 0);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Short... values) throws IndexOutOfBoundsException {
      for (Short b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Integer... values) throws IndexOutOfBoundsException {
      for (Integer b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Long... values) throws IndexOutOfBoundsException {
      for (Long b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Float... values) throws IndexOutOfBoundsException {
      for (Float b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Double... values) throws IndexOutOfBoundsException {
      for (Double b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(String... values) throws IndexOutOfBoundsException {
      for (String b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append this vector to the end of this vector
     * @param columnVector - Vector to be added
     * @return - The CudfColumn based on this builder values
     */
    public final Builder append(CudfColumnVector columnVector) {
      assert columnVector.rows <= (rows - currentIndex);
      assert columnVector.type == type;
      assert columnVector.offHeap.getHostData() != null;

      if (type == TypeId.STRING) {
        throw new UnsupportedOperationException(
            "Appending a string column vector client side is not currently supported");
      } else {
        data.copyFromHostBuffer(currentIndex * type.sizeInBytes, columnVector.offHeap.getHostData().data,
            0L,
            columnVector.getRowCount() * type.sizeInBytes);
      }

      if (columnVector.nullCount != 0) {
        if (valid == null) {
          allocateBitmaskAndSetDefaultValues();
        }
        //copy values from intCudfColumn to this
        BitVectorHelper.append(columnVector.offHeap.getHostData().valid, valid, currentIndex,
            columnVector.rows);
        nullCount += columnVector.nullCount;
      }
      currentIndex += columnVector.rows;
      return this;
    }

    private void allocateBitmaskAndSetDefaultValues() {
      long bitmaskSize = CudfColumn.getNativeValidPointerSize((int) rows);
      valid = HostMemoryBuffer.allocate(bitmaskSize);
      valid.setMemory(0, bitmaskSize, (byte) 0xFF);
    }

    /**
     * Append null value.
     */
    public final Builder appendNull() {
      setNullAt(currentIndex);
      currentIndex++;
      if (type == TypeId.STRING) {
        offsets.setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      }
      return this;
    }

    /**
     * Set a specific index to null.
     * @param index
     */
    public final Builder setNullAt(long index) {
      assert index < rows;

      // add null
      if (this.valid == null) {
        allocateBitmaskAndSetDefaultValues();
      }
      nullCount += BitVectorHelper.setNullAt(valid, index);
      return this;
    }

    /**
     * Finish and create the immutable CudfColumn.
     */
    public final CudfColumnVector build() {
      if (built) {
        throw new IllegalStateException("Cannot reuse a builder.");
      }
      CudfColumnVector cv = new CudfColumnVector(type,
          currentIndex, nullCount, data, valid, offsets);
      try {
        cv.ensureOnDevice();
        built = true;
      } finally {
        if (!built) {
          cv.close();
        }
      }
      return cv;
    }

    /**
     * Finish and create the immutable CudfColumn.
     */
    public final CudfColumnVector buildOnHost() {
      if (built) {
        throw new IllegalStateException("Cannot reuse a builder.");
      }
      CudfColumnVector cv = new CudfColumnVector(type,
          currentIndex, nullCount, data, valid, offsets);
      built = true;
      return cv;
    }

    /**
     * Close this builder and free memory if the CudfColumn wasn't generated. Verifies that
     * the data was released even in the case of an error.
     */
    @Override
    public final void close() {
      if (!built) {
        data.close();
        data = null;
        if (valid != null) {
          valid.close();
          valid = null;
        }
        if (offsets != null) {
          offsets.close();
          offsets = null;
        }
        built = true;
      }
    }

    @Override
    public String toString() {
      return "Builder{" +
          "data=" + data +
          "type=" + type +
          ", valid=" + valid +
          ", currentIndex=" + currentIndex +
          ", nullCount=" + nullCount +
          ", rows=" + rows +
          ", built=" + built +
          '}';
    }
  }
}