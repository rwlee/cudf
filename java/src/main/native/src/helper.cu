#include "helper.cuh" 

std::unique_ptr<cudf::column> make_string(std::vector<char>& strings, std::vector<cudf::size_type>& offsets) {

	return cudf::make_strings_column(strings, offsets);
	

}