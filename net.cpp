#include <vector>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

class Neuron {
public:

private:

};

class Net {
public:
	Net(const vector<unsigned>& graph);
	void forward(vector<double>& inputs);
	void backward(vector<double>& targets);
	void pred(vector<double>& preds);
private:
	vector<Layer> m_layers;

};

Net::Net(const vector<unsigned>& graph) {
	for (int i = 0; i < graph.size(); ++i) {
		m_layers.push_back(Layer());
		unsigned layer_size = graph[i];
		for (int j = 0; j < layer_size; ++j) {
			Neuron n;
			m_layers.back().push_back(n);
		}
	}
}

//int main() {
//	vector<unsigned> graph = { 3, 2, 1 };
//	Net net(graph);
//	vector<double> inputs, targets, preds;
//	net.forward(inputs);
//	net.backward(targets);
//	net.pred(preds);
//	return 0;
//}