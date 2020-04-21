#include "nn.h"
#include <cstdlib>

int main() {

	MLP mlp({ 2, 16, 16, 1 });
	Vec x_in;
	vector<double> x = { 0.4, 44.4 };
	for (auto n : x) x_in.push_back(Value(n));
	Vec out = mlp(x_in);
	out[0].grad = 1;

	//cout << mlp.layers[1].neurons[1].w[0];
	for (int i = 0; i < 10; ++i) {
		std::system("cls");
		out[0].backward();
		//cout << "OK" << mlp.layers[1].neurons[9].w[8];
		/*for(auto& p: mlp.parameters())
			p->data -= 0.001 * p->grad;*/
		for (auto& layer : mlp.layers) {
			for (auto& neuron : layer.neurons) {
				for (auto& p : neuron.w) {
					cout << p;
				}
				neuron.b.data -= 67;
			}
		}

		//for (auto& layer : mlp.layers) {
		//	for (auto& neuron : layer.neurons) {
		//		for (auto& p : neuron.w) {
		//			p.data -= 88;
		//		}
		//		neuron.b.data -= 67;
		//	}
		//}
		//
		
	}
	cout << mlp;
	cout << mlp.layers[1];
	cout << mlp.layers[1].neurons[0];
	cout << mlp.parameters().size();

	return 0;
}
