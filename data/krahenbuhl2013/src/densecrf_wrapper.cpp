#include <Eigen/Core>
// #include "densecrf.h"
#include "densecrf_wrapper.h"
#include <cstring>
#include <iostream>

DenseCRFWrapper::DenseCRFWrapper(int npixels, int nlabels)
: m_npixels(npixels), m_nlabels(nlabels) {
	// m_crf = new DenseCRF(npixels, nlabels);
}

DenseCRFWrapper::~DenseCRFWrapper() {
	// delete m_crf;
}

int DenseCRFWrapper::npixels() { return m_npixels; }
int DenseCRFWrapper::nlabels() { return m_nlabels; }

// std::vector< std::vector<float> >  DenseCRFWrapper::add_pairwise_energy(float* features_ptr,
// 			 int nfeatures, char * path, int path_length) {
// 	// std::cout << path_ << std::endl;
// 	std::string path_2(path);

// 	std::cout << "we are in dense crf" << std::endl;

// 	return m_crf->addPairwiseEnergy(
// 		Eigen::Map<const Eigen::MatrixXf>(features_ptr, nfeatures, m_npixels),
// 		path_2, DIAG_KERNEL,NORMALIZE_SYMMETRIC);

// }

std::vector< std::vector<float> >  DenseCRFWrapper::construct_sparse(float* features_ptr,
			 int nfeatures) {
	MatrixXf features = Eigen::Map<const Eigen::MatrixXf>(features_ptr, nfeatures, m_npixels);
	return lattice_.init(features);
	// return lattice_.compute_sparse_matrix();
}



// void DenseCRFWrapper::set_unary_energy(float* unary_costs_ptr) {
// 	m_crf->setUnaryEnergy(
// 		Eigen::Map<const Eigen::MatrixXf>(
// 			unary_costs_ptr, m_nlabels, m_npixels)
// 	);
// }

// void DenseCRFWrapper::map(int n_iters, int* labels) {
// 	VectorXs labels_vec = m_crf->map(n_iters);
// 	for (int i = 0; i < m_npixels; i ++)
// 		labels[i] = labels_vec(i);
// }
