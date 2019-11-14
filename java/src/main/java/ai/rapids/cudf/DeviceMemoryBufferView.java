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

/**
 * This class represents data in some form on the GPU. Closing this object will effectively release
 * the memory held by the buffer.  Note that because of pooling in RMM or reference counting if a
 * buffer is sliced it may not actually result in the memory being released.
 */
public class DeviceMemoryBufferView extends DeviceMemoryBuffer {
  private static final Logger log = LoggerFactory.getLogger(DeviceMemoryBufferView.class);

  DeviceMemoryBufferView(long address, long lengthInBytes) {
    super(address, lengthInBytes);
  }

  @Override
  public void close() {
    // don't close the buffer
  }
}