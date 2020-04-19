#pragma once
#include <cstdlib>
#include <random>
#include "value.h"

typedef vector<Value> Vec;

class Module {
	vector<Value> parameters();
	void zero_grad();
};


class Neuron : public Module {
public:
	Vec w;
	Value b;
	bool m_nonlin;
	Neuron(int nin, bool nonlin = true);
	Value operator()(Vec& x);
	vector<Value> parameters();
};


class Layer : public Module {
public:
	vector<Neuron> neurons;
	Layer(int nin, int nout);
	Vec operator()(Vec& x);
	vector<Value> parameters();
};


class MLP : public Module {
public:
	vector<Layer> layers;
	MLP(const vector<int>& lay_siz);
	Vec operator()(Vec x);
	vector<Value> parameters();
};

ostream& operator<<(std::ostream& os, Neuron& n);
ostream& operator<<(std::ostream& os, Layer& layer);
ostream& operator<<(std::ostream& os, MLP& mlp);