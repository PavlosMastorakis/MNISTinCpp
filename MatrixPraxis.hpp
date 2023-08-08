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

vector<double> VecKronProd(vector<double> a, vector<double> b) {
	vector<double> result;
	for (int i = 0; i < a.size(); i++) {
		result.push_back(a[i] * b[i]);
	}
	return result;
}

/* pair <vector<vector<double>>, vector<vector<double>>> backprop(pair<vector<double>, vector<double>> test) {
	vector<double> activation = test.first;
	vector<vector<double>> activations = { test.first }, zs;
	
}*/

vector<double> VecAdd(vector<double> A, vector<double> B) {
	vector<double> R;
	int n = A.size();
	for (int i = 0; i < n; i++) {
		R.push_back(A[i] + B[i]);
	}
	return R;
}

vector<double> cost_derivative(vector<double> output_activations, vector<double> y) {
	vector<double> temp_cost;
	for (int i = 0; i < y.size(); i++) {
		temp_cost.push_back(output_activations[i] - y[i]);
	}
	return temp_cost;
}
