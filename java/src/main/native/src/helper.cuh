#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp> 

std::unique_ptr<cudf::column> make_string(std::vector<char>& strings, std::vector<cudf::size_type>& offsets);