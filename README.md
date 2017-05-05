# MultiLayerPerceptron
Implementation of MLP

cuteboydot@gmail.com

- example : number classification

<br>
<img src="https://github.com/cuteboydot/MultiLayerPerceptron/blob/master/img/number_ex.JPG" />
</br>

- test result
<br>
<img src="https://github.com/cuteboydot/MultiLayerPerceptron/blob/master/img/test_result.JPG" />
</br>

- usage : train & test  
```cpp
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
```

- usage details : feed forward  

```cpp  
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
```    

- usage details : back propagation  

```cpp
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
```
