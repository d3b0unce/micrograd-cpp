#pragma once
#include <iostream>
#include <vector>

using namespace std;

class Value {
private:
	enum Back_mode { Add, Mul, Relu };

public:
	double data;
	double grad = 0;
	Value* in1;
	Value* in2;
	Value* in3;
	vector<Value*> prev;
	Back_mode back_mode;
	Value(double p_data);
	Value() = default;
	Value operator+(Value& other);
	Value operator*(Value& other);
	Value relu();
	void backward();
};

ostream& operator<<(std::ostream& os, Value& v);