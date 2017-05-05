
#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <float.h>

#include "MultiLayerPerceptron.h"

using namespace std;

void main()
{
	// MULTI LAYER PERCEPTRON EXAMPLE
	double ** input_data;
	double ** answer_data;
	double ** test_data;
	double * guess_data;

	// classify numbers
	#define SIZE_TRAIN 6
	#define SIZE_TEST 3
	#define SIZE_IN_ATTR 49
	#define SIZE_OUT_ATTR 3
	#define SIZE_HIDD 34

	double data1[SIZE_IN_ATTR] = {0,0,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0};
	double data2[SIZE_IN_ATTR] = {0,0,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,1,0,0};
	double data3[SIZE_IN_ATTR] = {0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0};
	double data4[SIZE_IN_ATTR] = {0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0};
	double data5[SIZE_IN_ATTR] = {0,0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,1,1,0,0};
	double data6[SIZE_IN_ATTR] = {0,1,1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,1,1,1,0};
	double answer1[SIZE_OUT_ATTR] = {0,0,1};
	double answer2[SIZE_OUT_ATTR] = {0,0,1};
	double answer3[SIZE_OUT_ATTR] = {0,1,0};
	double answer4[SIZE_OUT_ATTR] = {0,1,0};
	double answer5[SIZE_OUT_ATTR] = {1,0,0};
	double answer6[SIZE_OUT_ATTR] = {1,0,0};
	double test1[SIZE_IN_ATTR] = {0,0,1,1,1,0,0,1,1,1,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0};
	double test2[SIZE_IN_ATTR] = {0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0};
	double test3[SIZE_IN_ATTR] = {0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,0};

	// init train data
	input_data = new double*[SIZE_TRAIN];
	for(int a=0; a<SIZE_TRAIN; a++) {
		input_data[a] = new double[SIZE_IN_ATTR];
	}
	memcpy(input_data[0], data1, sizeof(data1));
	memcpy(input_data[1], data2, sizeof(data2));
	memcpy(input_data[2], data3, sizeof(data3));
	memcpy(input_data[3], data4, sizeof(data4));
	memcpy(input_data[4], data5, sizeof(data5));
	memcpy(input_data[5], data6, sizeof(data6));

	// init label list
	answer_data = new double*[SIZE_TRAIN];
	for(int a=0; a<SIZE_TRAIN; a++) {
		answer_data[a] = new double[SIZE_OUT_ATTR];
	}
	memcpy(answer_data[0], answer1, sizeof(answer1));
	memcpy(answer_data[1], answer2, sizeof(answer2));
	memcpy(answer_data[2], answer3, sizeof(answer3));
	memcpy(answer_data[3], answer4, sizeof(answer4));
	memcpy(answer_data[4], answer5, sizeof(answer5));
	memcpy(answer_data[5], answer6, sizeof(answer6));

	// init test data
	guess_data = new double[SIZE_OUT_ATTR];
	test_data = new double*[SIZE_TEST];
	for(int a=0; a<SIZE_TEST; a++) {
		test_data[a] = new double[SIZE_IN_ATTR];
	}
	memcpy(test_data[0], test1, sizeof(test1));
	memcpy(test_data[1], test2, sizeof(test2));
	memcpy(test_data[2], test3, sizeof(test3));

	// train
	CMultiLayerPerceptron * p = new CMultiLayerPerceptron();
	p->init(SIZE_TRAIN, SIZE_IN_ATTR, SIZE_HIDD, SIZE_OUT_ATTR);
	p->train(input_data, answer_data);

	// test
	for(int a=0; a<SIZE_TEST; a++) {
		p->classfication(test_data[a], guess_data);
		printf("[%d] guess:%.2f,%.2f,%.2f, answer:%.2f,%.2f,%.2f \n", a, 
			guess_data[0], guess_data[1], guess_data[2], 
			answer_data[a*2][0], answer_data[a*2][1], answer_data[a*2][2]);
	}
	//

	delete p;

	for(int a=0; a<SIZE_TRAIN; a++) {
		delete input_data[a];
		delete answer_data[a];
	}
	delete input_data;
	delete answer_data;

	for(int a=0; a<SIZE_TEST; a++) {
		delete test_data[a];
	}
	delete test_data;

	#undef SIZE_TRAIN
	#undef SIZE_TEST
	#undef SIZE_IN_ATTR
	#undef SIZE_OUT_ATTR
	#undef SIZE_HIDD

	getchar();
}

