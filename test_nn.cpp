#include "nn.h"

int main() {

	MLP mlp({ 2, 16, 16, 1 });
	Vec x_in;
	vector<double> x = { 0.4, 44.4 };
	for (auto n : x) x_in.push_back(Value(n));
	Vec out = mlp(x_in);
	out[0].grad = 1;
	for (int i = 0; i < 10; ++i) {
		out[0].backward();
		cout << out[0];
	}
	cout << mlp;
	cout << mlp.layers[1];
	cout << mlp.layers[1].neurons[0];
	cout << mlp.parameters().size();

	return 0;
}
