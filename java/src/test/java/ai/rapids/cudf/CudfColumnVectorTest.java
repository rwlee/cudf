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

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CudfColumnVectorTest extends CudfTestBase {

  // c = a * a - a
  static String ptx = "***(" +
"      .func _Z1fPii(" +
"        .param .b64 _Z1fPii_param_0," +
"        .param .b32 _Z1fPii_param_1" +
"  )" +
"  {" +
"        .reg .b32       %r<4>;" +
"        .reg .b64       %rd<3>;" +
"    ld.param.u64    %rd1, [_Z1fPii_param_0];" +
"    ld.param.u32    %r1, [_Z1fPii_param_1];" +
"    cvta.to.global.u64      %rd2, %rd1;" +
"    mul.lo.s32      %r2, %r1, %r1;" +
"    sub.s32         %r3, %r2, %r1;" +
"    st.global.u32   [%rd2], %r3;" +
"    ret;" +
"  }" +
")***";

  @Test
  void testTransformVector() {
    try (ColumnVector cv = ColumnVector.fromBoxedInts(2,3,null,4);
         ColumnVector cv1 = cv.transform(ptx, true);
         ColumnVector expected = ColumnVector.fromBoxedInts(2*2-2, 3*3-3, null, 4*4-4)) {
      for (int i = 0 ; i < cv1.getRowCount() ; i++) {
        cv1.ensureOnHost();
        assertEquals(expected.isNull(i), cv1.isNull(i));
        if (!expected.isNull(i)) {
          assertEquals(expected.getInt(i), cv1.getInt(i));
        }
      }
    }
  }

  @Test
  void testString() {
    try (ColumnVector cv = ColumnVector.fromStrings("d", "sd", "sd")) {
      cv.dropHostData();
      cv.ensureOnHost();
      for (int i = 0 ; i < cv.getRowCount() ; i++) {
        System.out.println(cv.getJavaString(i));
      }
    }
  }
}
