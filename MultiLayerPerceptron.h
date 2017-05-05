#pragma once

#define LOOP_MAX			10000
#define LEARNING_RATE		0.00037
#define ACCURACY_CRITERIA	95.0
#define MOMENTUM			0.9
#define LOGSIZE_MAX			1024

class CMultiLayerPerceptron
{
public:
	CMultiLayerPerceptron(void);
	~CMultiLayerPerceptron(void);

	void init(int nSizeRecord, int nSizeInput, int nSizeHidd, int nSizeOut);
	
	void train(double ** ppInputs, double ** ppAnswer);
	void classfication(double * pInputs, double * pGuess);
	void setlog(bool save);

	int m_nMaxLoop;
	int m_nRecord;
	int m_nInput, m_nHidden, m_nOutput;
	
	//neurons
	double* m_pInputNeurons;
	double* m_pHiddenNeurons;
	double* m_pOutputNeurons;

	//weights
	double** m_ppInputHiddenW;
	double** m_ppHiddenOutputW;

	// error gradient
	double* m_pHiddenError;
	double* m_pOutputError;

	// deltas
	double** m_ppInputHiddenD;
	double** m_ppHiddenOutputD;

	// etc param
	double m_dLearningRate;
	double m_dMomentum;

private:
	bool m_bSave;
	inline double sigmoid(double dVal);
	inline double sigmoid_inverse(double dVal);
	inline int clamp(double dVal);

	double activate(double dVal);
	void feedforward(double * pInputs);
	void backpropagation(double * pAnswer);
	double getaccuracy(double ** ppInputs, double ** ppAnswer);
	double getmse(double ** ppInputs, double ** ppAnswer);
	void savelog();
};

