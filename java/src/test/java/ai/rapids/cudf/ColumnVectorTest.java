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

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.IntStream;

import static ai.rapids.cudf.QuantileMethod.*;
import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class ColumnVectorTest extends CudfTestBase {

  public static final double DELTA = 0.0001;

  @Test
  void testDataMovement() {
    try (ColumnVector vec = ColumnVector.fromBoxedInts(1, 2, 3, 4, null, 6)) {
      assert vec.hasDeviceData();
      assert vec.hasHostData();
      vec.dropHostData();
      assert !vec.hasHostData();
      assert vec.hasDeviceData();
      vec.dropDeviceData();
      assert vec.hasHostData();
      assert !vec.hasDeviceData();
    }
  }

  @Test
  void testRefCountLeak() throws InterruptedException {
    assumeTrue(Boolean.getBoolean("ai.rapids.cudf.flaky-tests-enabled"));
    long expectedLeakCount = MemoryCleaner.leakCount.get() + 1;
    ColumnVector.fromInts(1, 2, 3);
    long maxTime = System.currentTimeMillis() + 10_000;
    long leakNow;
    do {
      System.gc();
      Thread.sleep(50);
      leakNow = MemoryCleaner.leakCount.get();
    } while (leakNow != expectedLeakCount && System.currentTimeMillis() < maxTime);
    assertEquals(expectedLeakCount, MemoryCleaner.leakCount.get());
  }

  @Test
  void testConcatTypeError() {
    try (ColumnVector v0 = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromFloats(5.0f, 6.0f)) {
      assertThrows(CudfException.class, () -> ColumnVector.concatenate(v0, v1));
    }
  }

  @Test
  void testConcatNoNulls() {
    try (ColumnVector v0 = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromInts(5, 6, 7);
         ColumnVector v2 = ColumnVector.fromInts(8, 9);
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2)) {
      v.ensureOnHost();
      assertEquals(9, v.getRowCount());
      assertFalse(v.hasNulls());
      assertFalse(v.hasValidityVector());
      for (int i = 0; i < 9; ++i) {
        assertEquals(i + 1, v.getInt(i), "at index " + i);
      }
    }
  }

  @Test
  void testConcatWithNulls() {
    try (ColumnVector v0 = ColumnVector.fromDoubles(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromDoubles(5, 6, 7);
         ColumnVector v2 = ColumnVector.fromBoxedDoubles(null, 9.0);
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2)) {
      v.ensureOnHost();
      assertEquals(9, v.getRowCount());
      assertTrue(v.hasNulls());
      assertTrue(v.hasValidityVector());
      for (int i = 0; i < 9; ++i) {
        if (i != 7) {
          assertEquals(i + 1, v.getDouble(i), "at index " + i);
        } else {
          assertTrue(v.isNull(i), "at index " + i);
        }
      }
    }
  }

  @Test
  void testConcatStrings() {
    try (ColumnVector v0 = ColumnVector.fromStrings("0","1","2",null);
         ColumnVector v1 = ColumnVector.fromStrings(null, "5", "6","7");
         ColumnVector expected = ColumnVector.fromStrings(
           "0","1","2",null,
           null,"5","6","7");
         ColumnVector v = ColumnVector.concatenate(v0, v1)) {
      assertColumnsAreEqual(v, expected);
    }
  }

  @Test
  void testConcaTimestamps() {
    try (ColumnVector v0 = ColumnVector.timestampMicroSecondsFromBoxedLongs(0L, 1L, 2L, null);
         ColumnVector v1 = ColumnVector.timestampMicroSecondsFromBoxedLongs(null, 5L, 6L, 7L);
         ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
           0L, 1L, 2L, null,
           null, 5L, 6L, 7L);
         ColumnVector v = ColumnVector.concatenate(v0, v1)) {
      assertColumnsAreEqual(v, expected);
    }
  }

  @Test
  void isNotNullTestEmptyColumn() {
    try (ColumnVector v = ColumnVector.fromBoxedInts();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void isNotNullTest() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(1, 2, null, 4, null, 6);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, false, true, false, true);
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void isNotNullTestAllNulls() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(null, null, null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false, false, false);
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void isNotNullTestAllNotNulls() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(1,2,3,4,5,6);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true, true, true);
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void isNullTest() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(1, 2, null, 4, null, 6);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, false, true, false);
         ColumnVector result = v.isNull()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void isNullTestEmptyColumn() {
    try (ColumnVector v = ColumnVector.fromBoxedInts();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         ColumnVector result = v.isNull()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testGetDeviceMemorySizeNonStrings() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(1, 2, 3, 4, 5, 6);
         ColumnVector v1 = ColumnVector.fromBoxedInts(1, 2, 3, null, null, 4, 5, 6)) {
      assertEquals(24, v0.getDeviceMemorySize()); // (6*4B)
      assertEquals(96, v1.getDeviceMemorySize()); // (8*4B) + 64B(for validity vector)
    }
  }

  @Test
  void testGetDeviceMemorySizeStrings() {
    try (ColumnVector v0 = ColumnVector.fromStrings("onetwothree", "four", "five");
         ColumnVector v1 = ColumnVector.fromStrings("onetwothree", "four", null, "five")) {
      assertEquals(80, v0.getDeviceMemorySize()); //32B + 24B + 24B
      assertEquals(168, v1.getDeviceMemorySize()); //32B + 24B + 24B + 24B + 64B(for validity vector)
    }
  }

  @Test
  void testFromScalarZeroRows() {
    for (DType type : DType.values()) {
      Scalar s = null;
      try {
        switch (type) {
        case BOOL8:
          s = Scalar.fromBool(true);
          break;
        case INT8:
          s = Scalar.fromByte((byte) 5);
          break;
        case INT16:
          s = Scalar.fromShort((short) 12345);
          break;
        case INT32:
          s = Scalar.fromInt(123456789);
          break;
        case INT64:
          s = Scalar.fromLong(1234567890123456789L);
          break;
        case FLOAT32:
          s = Scalar.fromFloat(1.2345f);
          break;
        case FLOAT64:
          s = Scalar.fromDouble(1.23456789);
          break;
        case TIMESTAMP_DAYS:
          s = Scalar.timestampDaysFromInt(12345);
          break;
        case TIMESTAMP_SECONDS:
        case TIMESTAMP_MILLISECONDS:
        case TIMESTAMP_MICROSECONDS:
        case TIMESTAMP_NANOSECONDS:
          s = Scalar.timestampFromLong(type, 1234567890123456789L);
          break;
        case STRING:
          s = Scalar.fromString("hello, world!");
          break;
        case EMPTY:
        case CATEGORY:
          continue;
        default:
          throw new IllegalArgumentException("Unexpected type: " + type);
        }

        try (ColumnVector c = ColumnVector.fromScalar(s, 0)) {
          assertEquals(type, c.getType());
          assertEquals(0, c.getRowCount());
          assertEquals(0, c.getNullCount());
        }
      } finally {
        if (s != null) {
          s.close();
        }
      }
    }
  }

  @Test
  void testFromScalar() {
    final int rowCount = 4;
    for (DType type : DType.values()) {
      Scalar s = null;
      ColumnVector expected = null;
      ColumnVector result = null;
      try {
        switch (type) {
        case BOOL8:
          s = Scalar.fromBool(true);
          expected = ColumnVector.fromBoxedBooleans(true, true, true, true);
          break;
        case INT8: {
          byte v = (byte) 5;
          s = Scalar.fromByte(v);
          expected = ColumnVector.fromBoxedBytes(v, v, v, v);
          break;
        }
        case INT16: {
          short v = (short) 12345;
          s = Scalar.fromShort(v);
          expected = ColumnVector.fromBoxedShorts((short) 12345, (short) 12345, (short) 12345, (short) 12345);
          break;
        }
        case INT32: {
          int v = 123456789;
          s = Scalar.fromInt(v);
          expected = ColumnVector.fromBoxedInts(v, v, v, v);
          break;
        }
        case INT64: {
          long v = 1234567890123456789L;
          s = Scalar.fromLong(v);
          expected = ColumnVector.fromBoxedLongs(v, v, v, v);
          break;
        }
        case FLOAT32: {
          float v = 1.2345f;
          s = Scalar.fromFloat(v);
          expected = ColumnVector.fromBoxedFloats(v, v, v, v);
          break;
        }
        case FLOAT64: {
          double v = 1.23456789;
          s = Scalar.fromDouble(v);
          expected = ColumnVector.fromBoxedDoubles(v, v, v, v);
          break;
        }
        case TIMESTAMP_DAYS: {
          int v = 12345;
          s = Scalar.timestampDaysFromInt(v);
          expected = ColumnVector.daysFromInts(v, v, v, v);
          break;
        }
        case TIMESTAMP_SECONDS: {
          long v = 1234567890123456789L;
          s = Scalar.timestampFromLong(type, v);
          expected = ColumnVector.timestampSecondsFromLongs(v, v, v, v);
          break;
        }
        case TIMESTAMP_MILLISECONDS: {
          long v = 1234567890123456789L;
          s = Scalar.timestampFromLong(type, v);
          expected = ColumnVector.timestampMilliSecondsFromLongs(v, v, v, v);
          break;
        }
        case TIMESTAMP_MICROSECONDS: {
          long v = 1234567890123456789L;
          s = Scalar.timestampFromLong(type, v);
          expected = ColumnVector.timestampMicroSecondsFromLongs(v, v, v, v);
          break;
        }
        case TIMESTAMP_NANOSECONDS: {
          long v = 1234567890123456789L;
          s = Scalar.timestampFromLong(type, v);
          expected = ColumnVector.timestampNanoSecondsFromLongs(v, v, v, v);
          break;
        }
        case STRING: {
          String v = "hello, world!";
          s = Scalar.fromString(v);
          expected = ColumnVector.fromStrings(v, v, v, v);
          break;
        }
        case EMPTY:
        case CATEGORY:
          continue;
        default:
          throw new IllegalArgumentException("Unexpected type: " + type);
        }

        result = ColumnVector.fromScalar(s, rowCount);
        assertColumnsAreEqual(expected, result);
      } finally {
        if (s != null) {
          s.close();
        }
        if (expected != null) {
          expected.close();
        }
        if (result != null) {
          result.close();
        }
      }
    }
  }

  @Test
  void testFromScalarNull() {
    final int rowCount = 4;
    for (DType type : DType.values()) {
      if (type == DType.EMPTY || type == DType.CATEGORY) {
        continue;
      }
      try (Scalar s = Scalar.fromNull(type);
           ColumnVector c = ColumnVector.fromScalar(s, rowCount)) {
        assertEquals(type, c.getType());
        assertEquals(rowCount, c.getRowCount());
        assertEquals(rowCount, c.getNullCount());
        c.ensureOnHost();
        for (int i = 0; i < rowCount; ++i) {
          assertTrue(c.isNull(i));
        }
      }
    }
  }

  @Test
  void testFromScalarNullByte() {
    int numNulls = 3000;
    try (Scalar s = Scalar.fromNull(DType.INT8);
         ColumnVector input = ColumnVector.fromScalar(s, numNulls)) {
      assertEquals(numNulls, input.getRowCount());
      assertEquals(input.getNullCount(), numNulls);
      input.ensureOnHost();
      for (int i = 0; i < numNulls; i++){
        assertTrue(input.isNull(i));
      }
    }
  }

  @Test
  void testReplaceEmptyColumn() {
    try (ColumnVector input = ColumnVector.fromBoxedBooleans();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         Scalar s = Scalar.fromBool(false);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullBoolsWithAllNulls() {
    try (ColumnVector input = ColumnVector.fromBoxedBooleans(null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false);
         Scalar s = Scalar.fromBool(false);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceSomeNullBools() {
    try (ColumnVector input = ColumnVector.fromBoxedBooleans(false, null, null, false);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false);
         Scalar s = Scalar.fromBool(true);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullIntegersWithAllNulls() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedInts(0, 0, 0, 0);
         Scalar s = Scalar.fromInt(0);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceSomeNullIntegers() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null);
         ColumnVector expected = ColumnVector.fromBoxedInts(1, 2, 999, 4, 999);
         Scalar s = Scalar.fromInt(999);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullsFailsOnTypeMismatch() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null);
         Scalar s = Scalar.fromBool(true)) {
      assertThrows(CudfException.class, () -> input.replaceNulls(s).close());
    }
  }

  @Test
  void testReplaceNullsWithNullScalar() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null);
         Scalar s = Scalar.fromNull(input.getType());
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(input, result);
    }
  }

  static QuantileMethod[] methods = {LINEAR, LOWER, HIGHER, MIDPOINT, NEAREST};
  static double[] quantiles = {0.0, 0.25, 0.33, 0.5, 1.0};

  @Test
  void testQuantilesOnIntegerInput() {
    int[] approxExpected = {-1, 1, 1, 2, 9};
    double[][] exactExpected = {
        {-1.0,   1.0,   1.0,   2.5,   9.0},  // LINEAR
        {  -1,     1,     1,     2,     9},  // LOWER
        {  -1,     1,     1,     3,     9},  // HIGHER
        {-1.0,   1.0,   1.0,   2.5,   9.0},  // MIDPOINT
        {  -1,     1,     1,     2,     9}}; // NEAREST

    try (ColumnVector cv = ColumnVector.fromBoxedInts(7, 0, 3, 4, 2, 1, -1, 1, 6, 9)) {
      // sorted: -1, 0, 1, 1, 2, 3, 4, 6, 7, 9
      for (int j = 0 ; j < quantiles.length ; j++) {
        Scalar result = cv.approxQuantile(quantiles[j]);
        assertEquals(approxExpected[j], result.getInt());

        for (int i = 0 ; i < methods.length ; i++) {
          result = cv.exactQuantile(methods[i], quantiles[j]);
          assertEquals(exactExpected[i][j], result.getDouble(), DELTA);
        }
      }
    }
  }

  @Test
  void testQuantilesOnDoubleInput() {
    double[] approxExpected = {-1.01, 0.8, 0.8, 2.13, 6.8};
    double[][] exactExpected = {
        {-1.01, 0.8, 0.9984, 2.13, 6.8},  // LINEAR
        {-1.01, 0.8,    0.8, 2.13, 6.8},  // LOWER
        {-1.01, 0.8,   1.11, 2.13, 6.8},  // HIGHER
        {-1.01, 0.8,  0.955, 2.13, 6.8},  // MIDPOINT
        {-1.01, 0.8,   1.11, 2.13, 6.8}}; // NEAREST

    try (ColumnVector cv = ColumnVector.fromBoxedDoubles(6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7)) {
      // sorted: -1.01, 0.15, 0.8, 1.11, 2.13, 3.4, 4.17, 5.7, 6.8
      for (int j = 0; j < quantiles.length ; j++) {
        Scalar result = cv.approxQuantile(quantiles[j]);
        assertEquals(approxExpected[j], result.getDouble(), DELTA);

        for (int i = 0 ; i < methods.length ; i++) {
          result = cv.exactQuantile(methods[i], quantiles[j]);
          assertEquals(exactExpected[i][j], result.getDouble(), DELTA);
        }
      }
    }
  }

  @Test
  void testSliceWithArray() {
    try(ColumnVector cv = ColumnVector.fromBoxedInts(10, 12, null, null, 18, 20, 22, 24, 26, 28)) {
      Integer[][] expectedSlice = {
          {12, null},
          {20, 22, 24, 26},
          {null, null},
          {}};

      ColumnVector[] slices = cv.slice(1, 3, 5, 9, 2, 4, 8, 8);

      try {
        for (int i = 0; i < slices.length; i++) {
          final int sliceIndex = i;
          ColumnVector slice = slices[sliceIndex];
          slice.ensureOnHost();
          assertEquals(expectedSlice[sliceIndex].length, slices[sliceIndex].getRowCount());
          IntStream.range(0, expectedSlice[sliceIndex].length).forEach(rowCount -> {
            if (expectedSlice[sliceIndex][rowCount] == null) {
              assertTrue(slices[sliceIndex].isNull(rowCount));
            } else {
              assertEquals(expectedSlice[sliceIndex][rowCount],
                  slices[sliceIndex].getInt(rowCount));
            }
          });
        }
        assertEquals(4, slices.length);
      } finally {
        for (int i = 0 ; i < slices.length ; i++) {
          if (slices[i] != null) {
            slices[i].close();
          }
        }
      }
    }
  }

  @Test
  void testSplitWithArray() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try(ColumnVector cv = ColumnVector.fromBoxedInts(10, 12, null, null, 18, 20, 22, 24, 26, 28)) {
      Integer[][] expectedData = {
          {10},
          {12, null},
          {null, 18},
          {20, 22, 24, 26},
          {28}};

      ColumnVector[] splits = cv.split(1, 3, 5, 9);
      try {
        assertEquals(expectedData.length, splits.length);
        for (int splitIndex = 0; splitIndex < splits.length; splitIndex++) {
          ColumnVector subVec = splits[splitIndex];
          subVec.ensureOnHost();
          assertEquals(expectedData[splitIndex].length, subVec.getRowCount());
          for (int subIndex = 0; subIndex < expectedData[splitIndex].length; subIndex++) {
            Integer expected = expectedData[splitIndex][subIndex];
            if (expected == null) {
              assertTrue(subVec.isNull(subIndex));
            } else {
              assertEquals(expected, subVec.getInt(subIndex));
            }
          }
        }
      } finally {
        for (int i = 0 ; i < splits.length ; i++) {
          if (splits[i] != null) {
            splits[i].close();
          }
        }
      }
    }
  }

  @Test
  void testWithOddSlices() {
    try (ColumnVector cv = ColumnVector.fromBoxedInts(10, 12, null, null, 18, 20, 22, 24, 26, 28)) {
      assertThrows(CudfException.class, () -> cv.slice(1, 3, 5, 9, 2, 4, 8));
    }
  }

  @Test
  void testSliceWithColumnVector() {
    try(ColumnVector cv = ColumnVector.fromBoxedInts(10, 12, null, null, 18, 20, 22, 24, 26, 28);
        ColumnVector indices = ColumnVector.fromInts(1, 3, 5, 9, 2, 4, 8, 8)) {
      Integer[][] expectedSlice = {
          {12, null},
          {20, 22, 24, 26},
          {null, null},
          {}};

      final ColumnVector[] slices = cv.slice(indices);
      try {
        for (int i = 0; i < slices.length; i++) {
          final int sliceIndex = i;
          ColumnVector slice = slices[sliceIndex];
          slice.ensureOnHost();
          assertEquals(expectedSlice[sliceIndex].length, slices[sliceIndex].getRowCount());
          IntStream.range(0, expectedSlice[sliceIndex].length).forEach(rowCount -> {
            if (expectedSlice[sliceIndex][rowCount] == null) {
              assertTrue(slices[sliceIndex].isNull(rowCount));
            } else {
              assertEquals(expectedSlice[sliceIndex][rowCount],
                  slices[sliceIndex].getInt(rowCount));
            }
          });
        }
        assertEquals(4, slices.length);
      } finally {
        for (int i = 0 ; i < slices.length ; i++) {
          if (slices[i] != null) {
            slices[i].close();
          }
        }
      }
    }
  }

  @Test
  void testAppendStrings() {
    //failing the test;
    assertTrue(false);
/*
    try (ColumnVector cv = ColumnVector.build(TypeId.STRING, 10, 0, (b) -> {
      b.append("123456789");
      b.append("1011121314151617181920");
      b.append("");
      b.appendNull();
    })) {
      assertEquals(4, cv.getRowCount());
      assertEquals("123456789", cv.getJavaString(0));
      assertEquals("1011121314151617181920", cv.getJavaString(1));
      assertEquals("", cv.getJavaString(2));
      assertTrue(cv.isNull(3));
    }
*/
  }

  @Test
  void testStringLengths() {
    try (ColumnVector cv = ColumnVector.fromStrings("1", "12", null, "123", "1234");
      ColumnVector lengths = cv.getLengths()) {
      lengths.ensureOnHost();
      assertEquals(5, lengths.getRowCount());
      for (int i = 0 ; i < lengths.getRowCount() ; i++) {
        if (cv.isNull(i)) {
          assertTrue(lengths.isNull(i));
        } else {
          assertEquals(cv.getJavaString(i).length(), lengths.getInt(i));
        }
      }
    }
  }

  @Test
  void testGetByteCount() {
    try (ColumnVector cv = ColumnVector.fromStrings("1", "12", "123", null, "1234");
         ColumnVector byteLengthVector = cv.getByteCount()) {
      byteLengthVector.ensureOnHost();
      assertEquals(5, byteLengthVector.getRowCount());
      for (int i = 0; i < byteLengthVector.getRowCount(); i++) {
        if (cv.isNull(i)) {
          assertTrue(byteLengthVector.isNull(i));
        } else {
          assertEquals(cv.getJavaString(i).length(), byteLengthVector.getInt(i));

        }
      }
    }
  }

  @Test
  void testEmptyStringColumnOpts() {
    try (ColumnVector cv = ColumnVector.fromStrings()) {
      try (ColumnVector len = cv.getLengths()) {
        assertEquals(0, len.getRowCount());
      }

      try (ColumnVector mask = ColumnVector.fromBoxedBooleans();
           Table input = new Table(cv);
           Table filtered = input.filter(mask)) {
        assertEquals(0, filtered.getColumn(0).getRowCount());
      }

      try (ColumnVector len = cv.getByteCount()) {
        assertEquals(0, len.getRowCount());
      }

      try (ColumnVector lower = cv.lower();
           ColumnVector upper = cv.upper()) {
        assertColumnsAreEqual(cv, lower);
        assertColumnsAreEqual(cv, upper);
      }
    }
  }

  @Test
  void testNVStringManipulationWithNulls() {
    // Special characters in order of usage, capital and small cyrillic koppa
    // Latin A with macron, and cyrillic komi de
    // \ud720 and \ud721 are UTF-8 characters without corresponding upper and lower characters
    try (ColumnVector v = ColumnVector.fromStrings("a", "B", "cd", "\u0480\u0481", "E\tf",
                                                   "g\nH", "IJ\"\u0100\u0101\u0500\u0501",
                                                   "kl m", "Nop1", "\\qRs2", null,
                                                   "3tuV\'", "wX4Yz", "\ud720\ud721");
         ColumnVector e_lower = ColumnVector.fromStrings("a", "b", "cd", "\u0481\u0481", "e\tf",
                                                         "g\nh", "ij\"\u0101\u0101\u0501\u0501",
                                                         "kl m", "nop1", "\\qrs2", null,
                                                         "3tuv\'", "wx4yz", "\ud720\ud721");
         ColumnVector e_upper = ColumnVector.fromStrings("A", "B", "CD", "\u0480\u0480", "E\tF",
                                                         "G\nH", "IJ\"\u0100\u0100\u0500\u0500",
                                                         "KL M", "NOP1", "\\QRS2", null,
                                                         "3TUV\'", "WX4YZ", "\ud720\ud721");
         ColumnVector lower = v.lower();
         ColumnVector upper = v.upper()) {
      assertColumnsAreEqual(lower, e_lower);
      assertColumnsAreEqual(upper, e_upper);
    }
  }

  @Test
  void testNVStringManipulation() {
    try (ColumnVector v = ColumnVector.fromStrings("a", "B", "cd", "\u0480\u0481", "E\tf",
                                                   "g\nH", "IJ\"\u0100\u0101\u0500\u0501",
                                                   "kl m", "Nop1", "\\qRs2", "3tuV\'",
                                                   "wX4Yz", "\ud720\ud721");
         ColumnVector e_lower = ColumnVector.fromStrings("a", "b", "cd", "\u0481\u0481", "e\tf",
                                                         "g\nh", "ij\"\u0101\u0101\u0501\u0501",
                                                         "kl m", "nop1", "\\qrs2", "3tuv\'",
                                                         "wx4yz", "\ud720\ud721");
         ColumnVector e_upper = ColumnVector.fromStrings("A", "B", "CD", "\u0480\u0480", "E\tF",
                                                         "G\nH", "IJ\"\u0100\u0100\u0500\u0500",
                                                         "KL M", "NOP1", "\\QRS2", "3TUV\'",
                                                         "WX4YZ", "\ud720\ud721");
         ColumnVector lower = v.lower();
         ColumnVector upper = v.upper()) {
      assertColumnsAreEqual(lower, e_lower);
      assertColumnsAreEqual(upper, e_upper);
    }
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4);
           ColumnVector lower = cv.lower()) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4);
           ColumnVector upper = cv.upper()) {}
    });
  }

  @Test
  void testWindowStatic() {
    WindowOptions v0 = WindowOptions.builder().windowSize(3).minPeriods(1).forwardWindow(1).
        aggType(AggregateOp.SUM).build();
    try (ColumnVector v1 = ColumnVector.fromBoxedInts(5, 4, 7, 6, 8);
         ColumnVector expected = ColumnVector.fromInts(9, 16, 22, 25, 21);
         ColumnVector result = v1.rollingWindow(v0)) {
      result.ensureOnHost();
      assertFalse(result.hasNulls());
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testWindowDynamic() {
    try (ColumnVector arraywindowCol = ColumnVector.fromBoxedInts(1, 2, 3, 1, 2)) {
         WindowOptions v0 = WindowOptions.builder().minPeriods(2).forwardWindow(2).
             windowCol(arraywindowCol).aggType(AggregateOp.SUM).build();
      try (ColumnVector v1 = ColumnVector.fromBoxedInts(5, 4, 7, 6, 8);
           ColumnVector expected = ColumnVector.fromInts(16, 22, 30, 14, 14);
           ColumnVector result = v1.rollingWindow(v0)) {
        result.ensureOnHost();
        assertColumnsAreEqual(result, expected);
      }
    }
  }

  @Test
  void testWindowThrowsException() {
    try (ColumnVector arraywindowCol = ColumnVector.fromBoxedInts(1, 2, 3 ,1, 1)) {
      assertThrows(IllegalArgumentException.class, () -> WindowOptions.builder().
              windowSize(3).minPeriods(3).forwardWindow(2).windowCol(arraywindowCol).
              aggType(AggregateOp.SUM).build());
    }
  }

  @Test
  void testFindAndReplaceAll() {
    try(ColumnVector vector = ColumnVector.fromInts(1, 4, 1, 5, 3, 3, 1, 2, 9, 8);
        ColumnVector oldValues = ColumnVector.fromInts(1, 4, 7); // 7 doesn't exist, nothing to replace
        ColumnVector replacedValues = ColumnVector.fromInts(7, 6, 1);
        ColumnVector expectedVector = ColumnVector.fromInts(7, 6, 7, 5, 3, 3, 7, 2, 9, 8);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
        assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllFloat() {
    try(ColumnVector vector = ColumnVector.fromFloats(1.0f, 4.2f, 1.3f, 5.7f, 3f, 3f, 1.0f, 2.6f, 0.9f, 8.3f);
        ColumnVector oldValues = ColumnVector.fromFloats(1.0f, 4.2f, 7); // 7 doesn't exist, nothing to replace
        ColumnVector replacedValues = ColumnVector.fromFloats(7.3f, 6.7f, 1.0f);
        ColumnVector expectedVector = ColumnVector.fromFloats(7.3f, 6.7f, 1.3f, 5.7f, 3f, 3f, 7.3f, 2.6f, 0.9f, 8.3f);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllTimeUnits() {
    try(ColumnVector vector = ColumnVector.timestampMicroSecondsFromLongs(1l, 1l, 2l, 8l);
        ColumnVector oldValues = ColumnVector.timestampMicroSecondsFromLongs(1l, 2l, 7l); // 7 dosn't exist, nothing to replace
        ColumnVector replacedValues = ColumnVector.timestampMicroSecondsFromLongs(9l, 4l, 0l);
        ColumnVector expectedVector = ColumnVector.timestampMicroSecondsFromLongs(9l, 9l, 4l, 8l);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllMixingTypes() {
    try(ColumnVector vector = ColumnVector.fromInts(1, 4, 1, 5, 3, 3, 1, 2, 9, 8);
        ColumnVector oldValues = ColumnVector.fromInts(1, 4, 7); // 7 doesn't exist, nothing to replace
        ColumnVector replacedValues = ColumnVector.fromFloats(7.0f, 6, 1)) {
      assertThrows(CudfException.class, () -> vector.findAndReplaceAll(oldValues, replacedValues));
    }
  }

  @Test
  void testFindAndReplaceAllStrings() {
    try(ColumnVector vector = ColumnVector.fromStrings("spark", "scala", "spark", "hello", "code");
        ColumnVector oldValues = ColumnVector.fromStrings("spark","code","hello");
        ColumnVector replacedValues = ColumnVector.fromStrings("sparked", "codec", "hi")) {
      assertThrows(CudfException.class, () -> vector.findAndReplaceAll(oldValues, replacedValues));
    }
  }

  @Test
  void testFindAndReplaceAllWithNull() {
    try(ColumnVector vector = ColumnVector.fromBoxedInts(1, 4, 1, 5, 3, 3, 1, null, 9, 8);
        ColumnVector oldValues = ColumnVector.fromBoxedInts(1, 4, 8);
        ColumnVector replacedValues = ColumnVector.fromBoxedInts(7, 6, null);
        ColumnVector expectedVector = ColumnVector.fromBoxedInts(7, 6, 7, 5, 3, 3, 7, null, 9, null);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllNulllWithValue() {
    // null values cannot be replaced using findAndReplaceAll();
    try(ColumnVector vector = ColumnVector.fromBoxedInts(1, 4, 1, 5, 3, 3, 1, null, 9, 8);
        ColumnVector oldValues = ColumnVector.fromBoxedInts(1, 4, null);
        ColumnVector replacedValues = ColumnVector.fromBoxedInts(7, 6, 8)) {
      assertThrows(CudfException.class, () -> vector.findAndReplaceAll(oldValues, replacedValues));
    }
  }

  @Test
  void testFindAndReplaceAllFloatNan() {
    // Float.NaN != Float.NaN therefore it cannot be replaced
    try(ColumnVector vector = ColumnVector.fromFloats(1.0f, 4.2f, 1.3f, 5.7f, 3f, 3f, 1.0f, 2.6f, Float.NaN, 8.3f);
        ColumnVector oldValues = ColumnVector.fromFloats(1.0f, 4.2f, Float.NaN);
        ColumnVector replacedValues = ColumnVector.fromFloats(7.3f, 6.7f, 0);
        ColumnVector expectedVector = ColumnVector.fromFloats(7.3f, 6.7f, 1.3f, 5.7f, 3f, 3f, 7.3f, 2.6f, Float.NaN, 8.3f);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllWithFloatNan() {
    try(ColumnVector vector = ColumnVector.fromFloats(1.0f, 4.2f, 1.3f, 5.7f, 3f, 3f, 1.0f, 2.6f, Float.NaN, 8.3f);
        ColumnVector oldValues = ColumnVector.fromFloats(1.0f, 4.2f, 8.3f);
        ColumnVector replacedValues = ColumnVector.fromFloats(7.3f, Float.NaN, 0);
        ColumnVector expectedVector = ColumnVector.fromFloats(7.3f, Float.NaN, 1.3f, 5.7f, 3f, 3f, 7.3f, 2.6f, Float.NaN, 0);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testCast() {
    int[] values = new int[]{1,3,4,5,2};
    long[] longValues = Arrays.stream(values).asLongStream().toArray();
    double[] doubleValues = Arrays.stream(values).asDoubleStream().toArray();
    byte[] byteValues = new byte[values.length];
    float[] floatValues = new float[values.length];
    short[] shortValues = new short[values.length];
    IntStream.range(0, values.length).forEach(i -> {
      byteValues[i] = (byte)values[i];
      floatValues[i] = (float)values[i];
      shortValues[i] = (short)values[i];
    });

    try (ColumnVector cv = ColumnVector.fromInts(values);
         ColumnVector expectedBytes = ColumnVector.fromBytes(byteValues);
         ColumnVector bytes = cv.asBytes();
         ColumnVector expectedFloats = ColumnVector.fromFloats(floatValues);
         ColumnVector floats = cv.asFloats();
         ColumnVector expectedDoubles = ColumnVector.fromDoubles(doubleValues);
         ColumnVector doubles = cv.asDoubles();
         ColumnVector expectedLongs = ColumnVector.fromLongs(longValues);
         ColumnVector longs = cv.asLongs();
         ColumnVector expectedShorts = ColumnVector.fromShorts(shortValues);
         ColumnVector shorts = cv.asShorts();
         ColumnVector expectedDays = ColumnVector.daysFromInts(values);
         ColumnVector days = cv.asTimestampDays();
         ColumnVector expectedUs = ColumnVector.timestampMicroSecondsFromLongs(longValues);
         ColumnVector us = cv.asTimestampMicroseconds();
         ColumnVector expectedNs = ColumnVector.timestampNanoSecondsFromLongs(longValues);
         ColumnVector ns = cv.asTimestampNanoseconds();
         ColumnVector expectedMs = ColumnVector.timestampMilliSecondsFromLongs(longValues);
         ColumnVector ms = cv.asTimestampMilliseconds();
         ColumnVector expectedS = ColumnVector.timestampSecondsFromLongs(longValues);
         ColumnVector s = cv.asTimestampSeconds();) {
      assertColumnsAreEqual(expectedBytes, bytes);
      assertColumnsAreEqual(expectedShorts, shorts);
      assertColumnsAreEqual(expectedLongs, longs);
      assertColumnsAreEqual(expectedDoubles, doubles);
      assertColumnsAreEqual(expectedFloats, floats);
      assertColumnsAreEqual(expectedDays, days);
      assertColumnsAreEqual(expectedUs, us);
      assertColumnsAreEqual(expectedMs, ms);
      assertColumnsAreEqual(expectedNs, ns);
      assertColumnsAreEqual(expectedS, s);
    }
  }
}
