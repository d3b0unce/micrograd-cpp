#include "value.h"

Value::Value(double p_data) {
	data = p_data;
}

Value Value::operator+(Value& other) {
	Value out(data + other.data);
	out.in1 = this;
	out.in2 = &other;
	out.back_mode = Add;
	return out;
}

Value Value::operator*(Value& other) {
	Value out(data * other.data);
	out.in1 = this;
	out.in2 = &other;
	out.back_mode = Mul;
	return out;
}

Value Value::relu() {
	double relu_data = data > 0 ? data : 0;
	Value out(relu_data);
	out.in3 = this;
	out.back_mode = Relu;
	return out;
}

void Value::backward() {
	switch (back_mode) {
	case Add:
		if (in1 != NULL && in2 != NULL) {
			in1->grad += grad;
			in2->grad += grad;
			in1->backward();
			in2->backward();
		}
		break;
	case Mul:
		if (in1 != NULL && in2 != NULL) {
			in1->grad += in2->data * grad;
			in2->grad += in1->data * grad; 
			in1->backward();
			in2->backward();
		}
		break;
	case Relu:
		if (in3 != NULL) {
			if (data > 0) in3->grad += grad;
			in3->backward();
		}
	}
	cout << *this;
}

ostream& operator<<(std::ostream& os, Value& v)
{
	return os << "Value(data=" << v.data << ", " << "grad=" << v.grad << ")\n";
}

//int main()
//{
//	Value v1(1), v2(1);
//	Value v3 = v1 * v2;
//
//	v1.grad = 1;
//	v2.grad = 2;
//	//v3 = v2 * v1;
//	Value v4 = v3.relu();
//	v4.grad = 1;
//
//	for (int i = 0; i < 100; ++i) {
//		//v3.rec_backward();
//		v4.backward();
//		v1.data -= 0.001 * v1.grad;
//		v2.data -= 0.001 * v2.grad;
//		v3.data -= 0.001 * v3.grad;
//		v4.data -= 0.001 * v4.grad;
//		cout << v1 << v2 << v3 << v4 << "\n";
//	}
//
//
//	return 0;
//}
