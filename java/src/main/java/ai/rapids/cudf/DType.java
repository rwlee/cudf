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

import java.util.EnumSet;

public enum DType {
  EMPTY(0, 0),
  INT8(1, 1),
  INT16(2, 2),
  INT32(4, 3),
  INT64(8, 4),
  FLOAT32(4, 5),
  FLOAT64(8, 6),
  /**
   * Byte wise true non-0/false 0.  In general true will be 1.
   */
  BOOL8(1, 7),
  /**
   * Days since the UNIX epoch
   */
  TIMESTAMP_DAYS(4, 8),
  /**
   * s since the UNIX epoch
   */
  TIMESTAMP_SECONDS(8, 9),
  /**
   * ms since the UNIX epoch
   */
  TIMESTAMP_MILLISECONDS(8, 10),
  /**
   * microseconds since the UNIX epoch
   */
  TIMESTAMP_MICROSECONDS(8, 11),
  /**
   * ns since the UNIX epoch
   */
  TIMESTAMP_NANOSECONDS(8, 12),
  CATEGORY(4, 13),
  STRING(0, 14);

  private static final DType[] TYPE_IDS = DType.values();
  final int sizeInBytes;
  final int nativeId;

  DType(int sizeInBytes, int nativeId) {
    this.sizeInBytes = sizeInBytes;
    this.nativeId = nativeId;
  }

  public boolean isTimestamp() {
    return TIMESTAMPS.contains(this);
  }

  /**
   * Returns true for timestamps with time level resolution, as opposed to day level resolution
   */
  public boolean hasTimeResolution() {
    return TIME_RESOLUTION.contains(this);
  }

  static DType fromNative(int nativeId) {
    for (DType type : TYPE_IDS) {
      if (type.nativeId == nativeId) {
        return type;
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a DType");
  }

  private static final EnumSet<DType> TIMESTAMPS = EnumSet.of(
      DType.TIMESTAMP_DAYS,
      DType.TIMESTAMP_SECONDS,
      DType.TIMESTAMP_MILLISECONDS,
      DType.TIMESTAMP_MICROSECONDS,
      DType.TIMESTAMP_NANOSECONDS);

  private static final EnumSet<DType> TIME_RESOLUTION = EnumSet.of(
      DType.TIMESTAMP_SECONDS,
      DType.TIMESTAMP_MILLISECONDS,
      DType.TIMESTAMP_MICROSECONDS,
      DType.TIMESTAMP_NANOSECONDS);
}