// Μπορεί να γίνει βελτίωση στη sigmoid (με void και reference ή με εφαρμογή σε διάνυσμα)
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <utility>
#include "mnist_reader.hpp"
#include "MatrixPraxis.hpp"

using namespace std;

default_random_engine generator;
normal_distribution<double> distribution(0.0, 1.0);

class Network {
private:
	int num_layers;
	vector<int> sizes;
	vector<vector<double>> biases;
	vector<vector<vector<double>>> weights;
public:
	//Constructors
	// Για το sizes
	// Για το biases: για ένα νευρωνικό δίκτυο με n layers, θέλουμε n-1 διανύσματα.
	// Κάθε ένα από τα διανύσματα θα έχει διάσταση, όσο το πλήθος των νευρώνων σε αυτό το layer.
	Network(vector<int> g_sizes)
	{
		sizes = g_sizes;
		num_layers = sizes.size();
		vector<int> subsizes = { sizes.begin() + 1, sizes.end() };
		int k(0);
		for (int i : subsizes) // Για τα biases, n-1 διανύσματα
		{
			vector<double> temp;
			for (int j = 0; j < i; j++) // Κάθε ένα με i (sizes[layer]) στοιχεία
			{
				temp.push_back(distribution(generator));
			}
			biases.push_back(temp);
			k++;
		}
		vector<int> subsizes2 = { sizes.begin(), sizes.end() - 1 };
		k = 0;
		for (int i : subsizes2) // Για τα weights, n-1 διανύσματα, όσα και τα layers - 1 (το τελευταίο δεν έχει)
		{
			vector<vector<double>> temp2;
			for (int t = 0; t < sizes[k + 1]; t++) // Κάθε ένα από αυτά περιλαμβάνει size[i] στοιχεία, όσοι και οι νευρώνες στο εν λόγω layer
			{
				vector<double> temp;
				for (int j = 0; j < i; j++) // Κάθε ένα από αυτά τα στοιχεία, περιέχει τόσες τιμές, όσοι είναι οι νευρώνες του επόμενου layer
				{
					temp.push_back(distribution(generator));
				}
				temp2.push_back(temp);
			}
			weights.push_back(temp2);
			k++;
		}
	}
	vector<vector<vector<double>>> getWeights() {
		return weights;
	}
	vector<vector<double>> getBiases() {
		return biases;
	}
	vector<int> getSizes() {
		return sizes;
	}
	int getNum_Layers() {
		return num_layers;
	}
};

int main()
{
	// ΔΙΑΒΑΖΕΙ ΤΟ DATASET
	auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>();
	// Κάνει tuple το training_images με το training_labels. Το tuple ονομάζεται training_data.
	vector< pair< vector<unsigned char>, unsigned char > > training_data;
	int n = dataset.training_images.size();
	for (int i = 0; i < n; i++) {
		training_data.push_back( { dataset.training_images[i], dataset.training_labels[i] } );
	}

	Network net = Network({2, 3, 1});
	/*
	vector<vector<double>> a; // a is a vector which contains a vector of activations for each layer
	vector<double> temp;
	for (int i = 0; i < net.getSizes()[0]; i++) {
		temp.push_back(distribution(generator));
	}
	a.push_back(temp); // a for the first layer (input layer)

	// Runs through the n-1 remaining layers, assigning values to them, filtered with the sigmoid function
	for (int i = 0; i < net.getNum_Layers() - 1; i++) {
		a.push_back(MatMultVec(net.getWeights()[i], a[i]));
		for (int j = 0; j < a[i + 1].size(); j++) {
			a[i + 1][j] = sigmoid(a[i + 1][j] + net.getBiases()[i][j]);
		}
	}
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[i].size(); j++) {
			cout << a[i][j] << " ";
		}
		cout << endl;
	}*/

	// Epoch training: given an epoch, we want the program to run (epoch) times. So, for loop, epoch times.
	// The epoch is given at the beggining of the program, along with other the other parameters.
	srand(time(0));
	int epoch = 2, mini_batch_size = 10;
	double eta = 3.0;
	for (int i = 0; i < epoch; i++) {
		random_shuffle(training_data.begin(), training_data.end());	
		vector< vector< pair< vector<unsigned char>, unsigned char > > > mini_batches;
		for (int k = 0; k < n; k += mini_batch_size) {
			mini_batches.push_back({ training_data.begin() + k,  training_data.begin() + k + mini_batch_size});
		}
		for (vector< pair< vector<unsigned char>, unsigned char > > mini_batch : mini_batches) {
			// Εδώ μπαίνει συνάρτηση (που στο βιβλίο ονομάζει update_mini_batch) με παραμέτρους mini_batch και eta (η)
		}
		cout << "Epoch " << i << " complete." << endl;
	}


	return 0;
}