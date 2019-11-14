#include "cudf/column/column.hpp"
#include "cudf/sorting.hpp"
#include "cudf/table/table_view.hpp"

#include "jni_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_TableView_createCudfTableView(JNIEnv *env, jclass clazz,
                                                                  jlongArray j_cudf_columns) {

  JNI_NULL_CHECK(env, j_cudf_columns, "columns are null", 0);

  try {
      cudf::jni::native_jpointerArray<cudf::column> n_cudf_columns(env, j_cudf_columns);

    std::vector<cudf::column_view> column_views(n_cudf_columns.size());
    for (int i = 0 ; i < n_cudf_columns.size() ; i++) {
        column_views[i] = n_cudf_columns[i]->view();
    }
    cudf::table_view* tv = new cudf::table_view(column_views);
    return reinterpret_cast<jlong>(tv);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_TableView_deleteCudfTableView(JNIEnv *env, jclass clazz,
                                                                  jlong j_cudf_table_view) {

  JNI_NULL_CHECK(env, j_cudf_table_view, "table view handle is null", );
  delete reinterpret_cast<cudf::table_view*>(j_cudf_table_view);
}



JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_TableView_sortedOrder(
    JNIEnv *env, jclass clazz, jlong j_input_table_view, jbooleanArray j_is_descending,
    jbooleanArray j_are_nulls_smallest) {

  JNI_NULL_CHECK(env, j_input_table_view, "input table is null", 0);
  JNI_NULL_CHECK(env, j_are_nulls_smallest, "null size passed as null", 0);
  try {
    const cudf::jni::native_jbooleanArray n_is_descending(env, j_is_descending);
    std::vector<cudf::order> order(n_is_descending.size());
    for (int i = 0; i < n_is_descending.size(); i++) {
      order[i] = n_is_descending[i] ? cudf::order::DESCENDING : cudf::order::ASCENDING;
    }
    const cudf::jni::native_jbooleanArray n_are_nulls_smallest(env, j_are_nulls_smallest);
    std::vector<cudf::null_order> null_order(n_are_nulls_smallest.size());
    for (int i = 0; i < n_are_nulls_smallest.size(); i++) {
      null_order[i] = n_are_nulls_smallest[i] ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
    }
    cudf::table_view *n_input_table_view = reinterpret_cast<cudf::table_view *>(j_input_table_view);
    auto sorted_col = cudf::experimental::sorted_order(*n_input_table_view, order, null_order);
    return reinterpret_cast<jlong>(sorted_col.release());
  }
  CATCH_STD(env, 0);
}
}
