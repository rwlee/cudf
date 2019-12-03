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

public class TableViewTest extends CudfTestBase {

  @Test
  void testSortedOrderColNullOrder() {
    try (Table t0 = new Table.TestBuilder()
        .column(5, 4, 3, null, 8, 5)
        .column(3, 4, 1, 4, 5, null)
        .column(9, 40, 70, null, 2, 10)
        .build();
         ColumnVector columnVector = t0.sortedOrder(Table.asc(0, true), Table.asc(1),Table.desc(2));
         ColumnVector expected = ColumnVector.fromInts(3, 2, 1, 0, 5, 4)) {
        columnVector.ensureOnHost();
      for (int i = 0 ; i < expected.getRowCount() ; i++) {
        assertEquals(expected.isNull(i), columnVector.isNull(i));
        if (!expected.isNull(i)) {
          assertEquals(expected.getInt(i), columnVector.getInt(i));
        }
      }
    }
  }
}
