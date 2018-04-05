// #include "densecrf.h"
#include "permutohedral.h"

class DenseCRFWrapper {
	public:
		DenseCRFWrapper(int npixels, int nlabels);
		virtual ~DenseCRFWrapper();
		// void set_unary_energy(float* unary_costs_ptr);
		// std::vector< std::vector<float> > add_pairwise_energy(float* features_ptr,
				 // int nfeatures, char * path, int path_length);
		std::vector< std::vector<float> >  construct_sparse(float* features_ptr,
					int nfeatures);
		// void map(int n_iters, int* result);

		int npixels();
		int nlabels();

	private:
		Permutohedral lattice_;
		// DenseCRF* m_crf;
		int m_npixels;
		int m_nlabels;
};
