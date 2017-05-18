//
// Created by Luc on 14/10/2015.
//

#ifndef NEURALNET_NEURON_H
#define NEURALNET_NEURON_H

#include <vector>
#include <cstdlib>

typedef std::vector<Neuron> Layer;

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned index);
    void feedFoward(const Layer &layer);
    void setOutputVal(double val) { outputVal_ = val; }
    double getOutputVal() const { return outputVal_; }
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    unsigned idx;
    double outputVal_;
    double gradient_;

    static double eta;
    static double alpha;

    std::vector<Connection> outputWeights_;

    static double randomWeight(){ return rand() / double(RAND_MAX); }
    static double transfertFn(double x);
    static double transfertFnDerivate(double x);

    double sumDOW(const Layer &nextLayer) const;

};



#endif //NEURALNET_NEURON_H
