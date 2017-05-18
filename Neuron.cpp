//
// Created by Luc on 14/10/2015.
//

#include "Neuron.h"
#include <cmath>

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned index) {
    for(unsigned c; c < numOutputs; c++) {
        outputWeights_.push_back(Connection());
        outputWeights_.back().weight = randomWeight();
    }

    idx = index;
}

void Neuron::feedFoward(const Layer &prevLayer) {
    double sum = 0.0;

    for(unsigned n = 0; n < prevLayer.size(); n++){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights_[idx].weight;
    }

    outputVal_ = Neuron::transfertFn(sum);
}

double Neuron::transfertFn(double x) {
    return tanh(x);
}

double Neuron::transfertFnDerivate(double x) {
    return 1 - x * x;
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - outputVal_;
    gradient_ = delta * Neuron::transfertFnDerivate(outputVal_);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    gradient_ = dow * Neuron::transfertFnDerivate(outputVal_);
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;

    for(unsigned n = 0; n < nextLayer.size() - 1; n++) {
        sum += outputWeights_[n].weight * nextLayer[n].gradient_;
    }
}


void Neuron::updateInputWeights(Layer &prevLayer) {
    for(unsigned n = 0; n < prevLayer.size(); n++) {
        Neuron &neuron = prevLayer[n];

        double oldDeltaWeight = neuron.outputWeights_[idx].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputVal() * gradient_ + alpha * oldDeltaWeight;

        neuron.outputWeights_[idx].deltaWeight = newDeltaWeight;
        neuron.outputWeights_[idx].weight += newDeltaWeight;
    }
}