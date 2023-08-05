#include <vector>

using namespace std;

inline double sigmoid(double z) {
	return (1 / (1 + exp(-z)));
}
inline double sigmoid_prime(double z) {
	return sigmoid(z) * (1 - sigmoid(z));
}

vector<double> MatMultVec(vector<vector<double>> A, vector<double> v) {
	vector<double> M;
	int r = A.size(), c = A[0].size();
	for (int i = 0; i < r; i++) {
		double z(0);
		for (int j = 0; j < c; j++) {
			z += A[i][j] * v[j];
		}
		M.push_back(z);
	}
	return M;
}
