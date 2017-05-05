#include "MultiLayerPerceptron.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

CMultiLayerPerceptron::CMultiLayerPerceptron(void)
{
	m_nMaxLoop = LOOP_MAX;
	m_nRecord = 0;
	m_nInput = 0;
	m_nHidden = 0;
	m_nOutput = 0;

	m_pInputNeurons = 0;
	m_pHiddenNeurons = 0;
	m_pOutputNeurons = 0;

	m_ppInputHiddenW = 0;
	m_ppHiddenOutputW = 0;

	m_bSave = false;
}

CMultiLayerPerceptron::~CMultiLayerPerceptron(void)
{
	if(m_pInputNeurons)		delete[] m_pInputNeurons;
	if(m_pHiddenNeurons)	delete[] m_pHiddenNeurons;
	if(m_pOutputNeurons)	delete[] m_pOutputNeurons;

	for (int a=0; a<=m_nInput; a++) {
		if(m_ppInputHiddenW[a])	
			delete[] m_ppInputHiddenW[a];
		if(m_ppInputHiddenD[a])	
			delete[] m_ppInputHiddenD[a];
	}
	if(m_ppInputHiddenW)	delete[] m_ppInputHiddenW;
	if(m_ppInputHiddenD)	delete[] m_ppInputHiddenD;

	for (int a=0; a<=m_nHidden; a++) {
		if(m_ppHiddenOutputW[a])	
			delete[] m_ppHiddenOutputW[a];
		if(m_ppHiddenOutputD[a])	
			delete[] m_ppHiddenOutputD[a];
	}
	if(m_ppHiddenOutputW)	delete[] m_ppHiddenOutputW;	
	if(m_ppHiddenOutputD)	delete[] m_ppHiddenOutputD;	

	if(m_pHiddenError)	delete[] m_pHiddenError;
	if(m_pOutputError)	delete[] m_pOutputError;
}

void CMultiLayerPerceptron::init(int nSizeRecord, int nSizeInput, int nSizeHidd, int nSizeOut)
{
	m_nRecord = nSizeRecord;
	m_nInput = nSizeInput;
	m_nHidden = nSizeHidd;
	m_nOutput = nSizeOut;

	m_dLearningRate = LEARNING_RATE;
	m_dMomentum = MOMENTUM;

	//create input neuron lists
	m_pInputNeurons = new(double[m_nInput+1]);
	for(int a=0; a<m_nInput; a++) {
		m_pInputNeurons[a] = 0;
	}
	//create input bias neuron
	m_pInputNeurons[m_nInput] = -1;

	//create hidden neuron lists / error gradient (hidden)
	m_pHiddenNeurons = new(double[m_nHidden+1]);
	for(int a=0; a<m_nHidden; a++) {
		m_pHiddenNeurons[a] = 0;
	}
	//create hidden bias neuron
	m_pHiddenNeurons[m_nHidden] = -1;

	//create output neuron lists / error gradient (output)
	m_pOutputNeurons = new(double[m_nOutput]);
	for (int a=0; a<m_nOutput; a++) {
		m_pOutputNeurons[a] = 0;
	}

	//create weight/deltas lists (include bias neuron weights)
	double rH = 1/sqrt((double) m_nInput);
	double rO = 1/sqrt((double) m_nInput);
	double wH = sqrt((double)(6.0) / (double)(m_nInput + m_nHidden));
	double wO = sqrt((double)(6.0) / (double)(m_nHidden + m_nInput));  
	int w1;
	double w2, w3;

	srand((unsigned int)time(0));
	m_ppInputHiddenW = new(double*[m_nInput+1]);
	m_ppInputHiddenD = new(double*[m_nInput+1]);
	for(int a=0; a<=m_nInput; a++ ) {
		m_ppInputHiddenW[a] = new (double[m_nHidden]);
		m_ppInputHiddenD[a] = new (double[m_nHidden]);
		for(int b=0; b<m_nHidden; b++) {
			//m_ppInputHiddenW[a][b] = (((double)(rand()%100)+1)/100  * 2 * rH) - rH;	

			w1 = (rand() % 100) + 1;
			w2 = (w1%2==0) ? (-1.0) : (1.0);
			w3 = (wH * w1) / 100 * w2;
			
			m_ppInputHiddenW[a][b] = w3;

			m_ppInputHiddenD[a][b] = 0;	
		}
	}

	m_ppHiddenOutputW = new(double*[m_nHidden+1]);
	m_ppHiddenOutputD = new(double*[m_nHidden+1]);
	for(int a=0; a<=m_nHidden; a++) {
		m_ppHiddenOutputW[a] = new (double[m_nOutput]);		
		m_ppHiddenOutputD[a] = new (double[m_nOutput]);
		for(int b=0; b<m_nOutput; b++) {
			//m_ppHiddenOutputW[a][b] = ( ( (double)(rand()%100)+1)/100 * 2 * rO ) - rO;

			w1 = (rand() % 100) + 1;
			w2 = (w1%2==0) ? (-1.0) : (1.0);
			w3 = (wO * w1) / 100 * w2;

			m_ppHiddenOutputW[a][b] = w3;

			m_ppHiddenOutputD[a][b] = 0;	
		}
	}	

	// create error gradient (hidden)
	m_pHiddenError = new(double[m_nHidden+1]);
	for(int a=0; a<=m_nHidden; a++) {
		m_pHiddenError[a] = 0;
	}

	// create error gradient (output)
	m_pOutputError = new(double[m_nOutput]);
	for(int a=0; a<m_nOutput; a++) {
		m_pOutputError[a] = 0;
	}
}

inline double CMultiLayerPerceptron::sigmoid(double dVal)
{
	return (1.0 / (1.0 + exp(-1.0 * dVal)));
}

inline double CMultiLayerPerceptron::sigmoid_inverse(double dVal)
{
	double dRet = sigmoid(dVal);
	return dRet * (1.0 - dRet);
}

double CMultiLayerPerceptron::activate(double dVal)
{
	double ret = 0.0;
	
	//ret = (tanh(dVal)+1.0)/2.0;
	ret = sigmoid(dVal);
	//ret = sigmoid_inverse(dVal);

	return ret;
}

inline int CMultiLayerPerceptron::clamp(double dVal)
{
	if(dVal < 0.1) return 0;
	else if(dVal > 0.9) return 1;
	else return -1;
}

void CMultiLayerPerceptron::setlog(bool save)
{
	m_bSave = save;
}

void CMultiLayerPerceptron::feedforward(double * pInputs)
{
	double dSum = 0;

	for(int a=0; a<m_nInput; a++) {
		m_pInputNeurons[a] = pInputs[a];
	}

	// Calculate Hidden Layer
	for(int b=0; b<m_nHidden; b++) {
		dSum = 0;
		m_pHiddenNeurons[b] = 0;

		// get weighted sum of pattern with bias (Input To Hidden)
		for(int a=0; a<=m_nInput; a++) {
			dSum += m_pInputNeurons[a] * m_ppInputHiddenW[a][b];
		}
		m_pHiddenNeurons[b] = activate(dSum);
	}

	// Calculate Output Layer
	for(int b=0; b<m_nOutput; b++) {
		dSum = 0;
		m_pOutputNeurons[b] = 0;

		// get weighted sum of pattern with bias (Hidden To Output)
		for(int a=0; a<=m_nHidden; a++) {
			dSum += m_pHiddenNeurons[a] * m_ppHiddenOutputW[a][b];
		}
		m_pOutputNeurons[b] = activate(dSum);
	}
}

void CMultiLayerPerceptron::backpropagation(double * pAnswer)
{
	// Modify deltas between hidden and output
	for(int b=0; b<m_nOutput; b++) {
		m_pOutputError[b] = m_pOutputNeurons[b] * (1.0 - m_pOutputNeurons[b]) * (pAnswer[b] - m_pOutputNeurons[b]);

		// get weighted sum of pattern with bias (Hidden To Output)
		for(int a=0; a<=m_nHidden; a++) {
			m_ppHiddenOutputD[a][b] = (LEARNING_RATE * m_pHiddenNeurons[a] * m_pOutputError[b]) + m_dMomentum * m_ppHiddenOutputD[a][b];
		}
	}

	// Modify deltas between input and hidden
	for(int b=0; b<m_nHidden; b++) {
		double sum = 0.0;

		for(int c=0; c<m_nOutput; c++) {
			sum += m_ppHiddenOutputW[b][c] * m_pOutputError[c];
		}
		m_pHiddenError[b] = m_pHiddenNeurons[b] * (1.0 - m_pHiddenNeurons[b]) * sum;

		// get weighted sum of pattern with bias (input To hidden)
		for(int a=0; a<=m_nInput; a++) {
			m_ppInputHiddenD[a][b] = (LEARNING_RATE * m_pInputNeurons[a] * m_pHiddenError[b]) + m_dMomentum * m_ppInputHiddenD[a][b];
		}
	}

	// update weight between input and hidden
	for(int a=0; a<=m_nInput; a++) {
		for(int b=0; b<m_nHidden; b++) {
			m_ppInputHiddenW[a][b] += m_ppInputHiddenD[a][b];
		}
	}

	// update weight between hidden and output
	for(int a=0; a<=m_nHidden; a++) {
		for(int b=0; b<m_nOutput; b++) {
			m_ppHiddenOutputW[a][b] += m_ppHiddenOutputD[a][b];
		}
	}
}

double CMultiLayerPerceptron::getaccuracy(double ** ppInputs, double ** ppAnswer)
{
	int incorrect = 0;
	for(int a=0; a<m_nRecord; a++) {
		feedforward(ppInputs[a]);
		
		for(int b=0; b<m_nOutput; b++) {
			if(clamp(m_pOutputNeurons[b]) != ppAnswer[a][b]) {
				incorrect++;
				break;
			}
		}
	}

	return 100 - (double)(incorrect*100) / (double)m_nRecord;
}

double CMultiLayerPerceptron::getmse(double ** ppInputs, double ** ppAnswer)
{
	double mse = 0.0;

	for(int a=0; a<m_nRecord; a++) {
		feedforward(ppInputs[a]);

		for(int b=0; b<m_nOutput; b++) {
			mse += pow((double)(m_pOutputNeurons[b] - ppAnswer[a][b]), 2);
		}
	}
	
	return mse / (m_nOutput * m_nRecord);
}

void CMultiLayerPerceptron::train(double ** ppInputs, double ** ppAnswer)
{
	int loop = 0;
	double accuracy = 0.0, accuracy2 = 0.0;
	double mse = 0.0;
	double diff;

	FILE *logfile = NULL;
	char * logmsg = new char[LOGSIZE_MAX];
	int loglen = 0;

	if(m_bSave) {
		char file[128];
		sprintf(file, "log_%d.txt", m_nHidden);
		logfile = fopen(file,"w");

		if(logfile) {
			m_bSave = true;
		} else {
			m_bSave = false;
		}
	}

	if(m_bSave) {
		printf("\nNeural Network ... start ...\n");
		loglen = sprintf(logmsg,"Neural Network ... start ...\n");
		fprintf(logfile,logmsg);

		printf("hidden neurons : %d, target accuracy : %d \n", m_nHidden, (int)ACCURACY_CRITERIA);
		loglen = sprintf(logmsg, "hidden neurons : %d, target accuracy : %d \n", m_nHidden, (int)ACCURACY_CRITERIA);
		fprintf(logfile,logmsg);
	}

	time_t start, end;
	time(&start);
	
	while(loop < m_nMaxLoop && accuracy < ACCURACY_CRITERIA) {
		for(int a=0; a<m_nRecord; a++) {
			feedforward(ppInputs[a]);
			backpropagation(ppAnswer[a]);
		}

		loop++;

		// optional func
		mse = getmse(ppInputs, ppAnswer);
		accuracy = getaccuracy(ppInputs, ppAnswer);

		if(loop % 500 == 0) {
			printf("[%d] accuracy = %.2f, mse = %.5f \n", loop, accuracy, mse);
			accuracy2 = accuracy;

			if(m_bSave) {
				loglen = sprintf(logmsg, "[%d], accuracy = %.2f, mse = %.5f\n", loop, accuracy, mse);
				fprintf(logfile,logmsg);
			}
		}

		if(loop == m_nMaxLoop-1) {
			printf("Max loop : %d \n", loop);
			accuracy2 = accuracy;
		}
	}
	time(&end);

	// print result..
	printf("Neural Network ... learning complete! ... \n");
	if(m_bSave) {
		loglen = sprintf(logmsg, "Learning complete! \n\n");
		fprintf(logfile,logmsg);
	}

	diff = difftime(end, start);
	printf("Elapsed time = %.2lf sec \n", diff);
	if(m_bSave) {
		loglen = sprintf(logmsg, "Elapsed time = %.2lf sec \n", diff);
		fprintf(logfile,logmsg);
	}

	for(int a=0; a<=m_nInput; a++) {
		//printf("i[%d] : ", a);
		if(m_bSave && loglen < LOGSIZE_MAX-1024) {
			loglen = sprintf(logmsg, "i[%d] : ", a);
			fprintf(logfile,logmsg);
		}

		for(int b=0; b<m_nHidden; b++) {
			//printf("h[%d]=%.3f  ", b, m_ppInputHiddenW[a][b]);
			if(m_bSave && loglen < LOGSIZE_MAX-1024) {
				loglen += sprintf(logmsg+loglen,"h[%d]=%.3f  ", b, m_ppInputHiddenW[a][b]);
			}
		}
		//printf("\n");
		if(m_bSave && loglen < LOGSIZE_MAX-1024) {
			loglen += sprintf(logmsg+loglen,"\n");
		}
	}

	printf("\n");
	if(m_bSave && loglen < LOGSIZE_MAX-1024) {
		loglen += sprintf(logmsg+loglen,"\n");
	}

	for(int a=0; a<=m_nHidden; a++) {
		//printf("h[%d] : ", a);
		if(m_bSave && loglen < LOGSIZE_MAX-1024) {
			loglen += sprintf(logmsg+loglen,"h[%d] : ", a);
		}

		for(int b=0; b<m_nOutput; b++) {
			//printf("o[%d]=%.3f  ", b, m_ppHiddenOutputW[a][b]);
			if(m_bSave && loglen < LOGSIZE_MAX-1024) {
				loglen += sprintf(logmsg+loglen,"o[%d]=%.3f  ", b, m_ppHiddenOutputW[a][b]);
			}
		}
		//printf("\n");
		if(m_bSave && loglen < LOGSIZE_MAX-1024) {
			loglen += sprintf(logmsg+loglen,"\n");
		}
	}

	if(m_bSave) {
		if(logfile) {
			fclose(logfile);
		}
	}

	printf("\n");
}

void CMultiLayerPerceptron::classfication(double * pInputs, double * pGuess)
{
	feedforward(pInputs);
	for(int a=0; a<m_nOutput; a++) {
		pGuess[a] = m_pOutputNeurons[a];
		//pGuess[a] = clamp(pGuess[a]);
	}
}
