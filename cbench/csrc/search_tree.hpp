#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>

template <typename T>
class SearchTreeNode {

public:
    SearchTreeNode() = default;

    SearchTreeNode(T data) : data(data) {};

    // SearchTreeNode(py::array_t<T> ndarray) {
    //     const auto ndim = ndarray.ndim();
    //     ssize_t dim_size = ndarray.size();
    //     for (ssize_t d=0; d<ndim; d++) {
    //         sub_nodes.resize(ndarray.shape(d))
    //         dim_size = dim_size / ndarray.shape(d);
    //     }
    // };

    SearchTreeNode(std::vector<T> flat, std::vector<size_t> dims) {
        if (dims.size() == 0) {
            data = flat[0];
            return;
        }

        // TODO: debug only?
        size_t total_size = 1;
        for (size_t dim : dims) total_size *= dim;
        assert(flat.size() == total_size);
        // printf("Create tree from %u length vector, current dim %u\n", flat.size(), dims[0]);

        size_t dim_size = flat.size() / dims[0];
        std::vector<size_t> sub_dim(dims.begin() + 1, dims.end());
        // sub_nodes.resize(dims[0]);
        for (size_t i=0; i<dims[0]; i++) {
            if (dims.size() == 1) {
                sub_nodes.emplace_back(flat[i]);
            }
            else {
                sub_nodes.emplace_back(
                    std::vector<T>(flat.begin() + i*dim_size, flat.begin() + (i+1)*dim_size), 
                    sub_dim
                );
            }
        }
    };

    // T index(size_t i, size_t... indexes) {
    //     if (i < sub_nodes.size()) {
    //         return sub_nodes[i].index(indexes...);
    //     }
    // };

    SearchTreeNode<T>* index_node(size_t i) {
        if (i >= sub_nodes.size()) throw std::out_of_range("index_node(i)");
        // printf("sub_nodes[i].sub_nodes.size() = %u\n", i, sub_nodes[i].sub_nodes.size());
        return &sub_nodes[i];
    };


    SearchTreeNode<T>* index_node(std::vector<size_t> indexes) {
        SearchTreeNode<T>* cur_node = this;
        for (size_t i : indexes ) {
            // printf("%u / %u\n", i, sub_nodes.size());
            if (i >= sub_nodes.size()) throw std::out_of_range("index_node(indexes)");
            cur_node = &(cur_node->sub_nodes[i]);
        }
        return cur_node;
    };

    T index_data(size_t i) {
        return index_node(i)->data;
    };

    T index_data(std::vector<size_t> indexes) {
        return index_node(indexes)->data;
    };


protected:
    T data;
    std::vector<SearchTreeNode<T>> sub_nodes;
};