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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include "helper.cuh"
#include "jni_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CudfColumn_makeNumericCudfColumn(
    JNIEnv *env, jobject j_object, jint j_type, jint j_size, jint j_mask_state) {

  JNI_NULL_CHECK(env, j_type, "type id is null", 0);
  JNI_NULL_CHECK(env, j_size, "size is null", 0);

  try {
    cudf::type_id n_type = static_cast<cudf::type_id>(j_type);
    std::unique_ptr<cudf::data_type> n_data_type(new cudf::data_type(n_type));
    cudf::size_type n_size = static_cast<cudf::size_type>(j_size);
    cudf::mask_state n_mask_state = static_cast<cudf::mask_state>(j_mask_state);
    std::unique_ptr<cudf::column> column(
        cudf::make_numeric_column(*n_data_type.get(), n_size, n_mask_state));
    return reinterpret_cast<jlong>(column.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CudfColumn_makeStringCudfColumn(
    JNIEnv *env, jobject j_object, jlong j_char_data, jlong j_offset_data, jlong j_valid_data,
    jint j_null_count, jint size) {

  JNI_NULL_CHECK(env, j_char_data, "char data is null", 0);
  JNI_NULL_CHECK(env, j_offset_data, "offset is null", 0);

  try {
    uint32_t *host_offsets = reinterpret_cast<uint32_t *>(j_offset_data);
    char *n_char_data = reinterpret_cast<char *>(j_char_data);
    uint32_t n_data_size = host_offsets[size];
    std::vector<char> strings(n_char_data, n_char_data + n_data_size);
    std::vector<cudf::size_type> offsets(host_offsets, host_offsets + size + 1);

    std::unique_ptr<cudf::column> column = make_string(strings, offsets);
    cudf::strings::print(cudf::strings_column_view(*column.get()));
    return reinterpret_cast<jlong>(column.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_CudfColumn_getNativeTypeId(JNIEnv *env, jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    return column->type().id();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_CudfColumn_getNativeRowCount(JNIEnv *env,
                                                                        jobject j_object,
                                                                        jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    return static_cast<jint>(column->size());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_CudfColumn_getNativeNullCount(JNIEnv *env,
                                                                         jobject j_object,
                                                                         jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    return static_cast<jint>(column->null_count());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_CudfColumn_deleteCudfColumn(JNIEnv *env,
                                                                       jobject j_object,
                                                                       jlong handle) {
  JNI_NULL_CHECK(env, handle, "column handle is null", );
  delete reinterpret_cast<cudf::column *>(handle);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CudfColumn_getNativeDataPointer(JNIEnv *env,
                                                                            jobject j_object,
                                                                            jlong handle) {
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    return reinterpret_cast<jlong>(static_cast<void *>(column->mutable_view().data<char>()));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CudfColumn_getNativeValidPointer(JNIEnv *env,
                                                                             jobject j_object,
                                                                             jlong handle) {
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    return reinterpret_cast<jlong>(column->mutable_view().null_mask());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CudfColumn_getNativeValidPointerSize(JNIEnv *env,
                                                                                 jobject j_object,
                                                                                 jint size) {
  try {
    return static_cast<jlong>(cudf::bitmask_allocation_size_bytes(size));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CudfColumn_transform(JNIEnv *env, jobject j_object,
                                                                 jlong handle, jstring j_udf,
                                                                 jboolean j_is_ptx) {
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    cudf::jni::native_jstring n_j_udf(env, j_udf);
    std::string n_udf(n_j_udf.get());
    std::unique_ptr<cudf::column> result = cudf::experimental::transform(
        column->view(), n_udf, cudf::data_type(cudf::INT32), j_is_ptx);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_CudfColumn_getStringDataAndOffsets(JNIEnv *env,
                                                                               jobject j_object,
                                                                               jlong handle) {
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    cudf::strings_column_view strings_column(column->view());
    cudf::column_view characters = strings_column.chars();
    cudf::column_view offsets = strings_column.offsets();

    cudf::jni::native_jlongArray ret(env, 4);
    ret[0] = reinterpret_cast<jlong>(static_cast<const void *>(characters.data<char>()));
    ret[1] = characters.size();
    ret[2] = reinterpret_cast<jlong>(static_cast<const void *>(offsets.data<int32_t>()));
    ret[3] = sizeof(int) * offsets.size();
    return ret.get_jArray();
  }
  CATCH_STD(env, 0);
}
} // extern C
