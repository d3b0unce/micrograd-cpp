#pragma once
#include <iostream>
#include <vector>

using namespace std;

class Value {
private:
	enum Back_mode { Add, Mul, Relu, None };

public:
	double data;
	double grad;
	Value* in1;
	Value* in2;
	Value* in3;
	vector<Value*> prev;
	Back_mode back_mode;
	Value(const double p_data);
	Value(const Value& other);
	void copyValue(Value** cur_val, const Value* other_val);
	Value();
	Value operator+(Value& other);
	Value operator*(Value& other);
	Value operator=(const Value& other);
	Value relu();
	void backward();
};

ostream& operator<<(std::ostream& os, Value& v);