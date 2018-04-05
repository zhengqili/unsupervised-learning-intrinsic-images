# distutils: sources = src/densecrf_wrapper.cpp

cimport numpy as np
from libcpp.vector cimport vector

import sys

cdef extern from "include/densecrf_wrapper.h":
    cdef cppclass DenseCRFWrapper:
        DenseCRFWrapper(int, int) except +
        # void set_unary_energy(float*)
        # vector[vector[float]] add_pairwise_energy(float*, int, char *, int)
        vector[vector[float]] construct_sparse(float *, int)
        # void map(int, int*)
        int npixels()
        int nlabels()

cdef class DenseCRF:
    cdef DenseCRFWrapper *thisptr

    def __cinit__(self, int npixels, int nlabels):
        self.thisptr = new DenseCRFWrapper(npixels, nlabels)

    def __dealloc__(self):
        del self.thisptr

    # def set_unary_energy(self, float[:, ::1] unary_costs):
    #     if (unary_costs.shape[0] != self.thisptr.npixels() or
    #             unary_costs.shape[1] != self.thisptr.nlabels()):
    #         raise ValueError("Invalid unary_costs shape")

    #     self.thisptr.set_unary_energy(&unary_costs[0, 0])

    def construct_sparse(self, float[:, ::1] features):
        if (features.shape[0] != self.thisptr.npixels()):
            raise ValueError("Invalid features shape")

        return self.thisptr.construct_sparse(&features[0,0], features.shape[1])

    # def add_pairwise_energy(self, float[:, ::1] pairwise_costs,
    #                         float[:, ::1] features, str path):

    #     # print(path)
    #     path_encode = path.encode('utf-8')
    #     # sys.exit()
    #     # if (pairwise_costs.shape[0] != self.thisptr.nlabels() or
    #             # pairwise_costs.shape[1] != self.thisptr.nlabels()):
    #         # raise ValueError("Invalid pairwise_costs shape")

    #     if (features.shape[0] != self.thisptr.npixels()):
    #         raise ValueError("Invalid features shape")
    #     # print("we are here 1")
    #     return self.thisptr.add_pairwise_energy(
    #         &features[0, 0],
    #         features.shape[1], path_encode, len(path))

    # def map(self, int n_iters=10):
        # import numpy as np
        # labels = np.empty(self.thisptr.npixels(), dtype=np.int32)
        # cdef int[::1] labels_view = labels
        # self.thisptr.map(n_iters, &labels_view[0])
        # return labels
