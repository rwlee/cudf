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
    try (TableView t0 = new TableView.TestBuilder()
        .column(5, 4, 3, null, 8, 5)
        .column("d", "e", "a", null, "k", "d")
        .column(9, 40, 70, null, 2, 10)
        .build();
         CudfColumnVector columnVector = t0.sortedOrder(TableView.asc(0, true), TableView.asc(1),TableView.desc(2));
         ColumnVector expected = ColumnVector.fromInts(3, 2, 1, 5, 0, 4)) {
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
