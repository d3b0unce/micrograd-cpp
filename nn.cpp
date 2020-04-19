#include "nn.h"

default_random_engine gen(42);
uniform_real_distribution<double> dis(-1.0, 1.0);


vector<Value> Module::parameters() {
	return {};
}
void Module::zero_grad() {
	for (Value p : parameters()) p.grad = 0;
}


Neuron::Neuron(int nin, bool nonlin) {
	for (int i = 0; i < nin; ++i) {
		w.push_back(Value(dis(gen)));
	}
	b = Value(0);
	m_nonlin = nonlin;
}

Value Neuron::operator()(Vec& x) {
	Value act(0);
	for (int i = 0; i < x.size(); ++i) {
		Value sum = w[i] * x[i];
		Value old_act = act;
		act = old_act + sum;
	}
	act = act + b;
	return m_nonlin ? act.relu() : act;
}

vector<Value> Neuron::parameters() {
	vector<Value> params;
	for (Value& v : w) params.push_back(v);
	params.push_back(b);
	return params;
}

ostream& operator<<(std::ostream& os, Neuron& n) {
	return os << "Non-linearity enabled: " << n.m_nonlin << endl;
}


Layer::Layer(int nin, int nout) {
	for (int i = 0; i < nout; ++i)
		neurons.push_back(Neuron(nin));
}

Vec Layer::operator()(Vec& x) {
	Vec out;
	for (int i = 0; i < neurons.size(); ++i) {
		out.push_back(neurons[i](x));
	}
	return out;
}

vector<Value> Layer::parameters() {
	vector<Value> params;
	for (Neuron& n : neurons) {
		for (Value& v : n.parameters()) {
			params.push_back(v);
		}
	}
	return params;
}

ostream& operator<<(std::ostream& os, Layer& layer) {
	return os << "Layer size: " << layer.neurons.size() << endl;
}


MLP::MLP(const vector<int>& lay_siz) {
	for (int i = 0; i < lay_siz.size() - 1; ++i) {
		layers.push_back(Layer(lay_siz[i], lay_siz[i + 1]));
	}
}

Vec MLP::operator()(Vec x) {
	for (auto layer : layers) {
		x = layer(x);
	}
	return x;
}

vector<Value> MLP::parameters() {
	vector<Value> params;
	for (Layer& layer : layers) {
		for (Value& v : layer.parameters()) {
			params.push_back(v);
		}
	}
	return params;
}

ostream& operator<<(std::ostream& os, MLP& mlp) {
	return os << "Numer of layers: " << mlp.layers.size() << endl;
}




