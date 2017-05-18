//
// Created by Luc on 14/10/2015.
//

#include "Network.h"
#include <cassert>

Network::Network(const std::vector<double> &topology) {
    unsigned nbLayers = topology.size();

    // Creating layers given by topology
    for(unsigned l = 0; l < nbLayers; l++) {
        netLayers_.push_back(Layer());

        //Number of outputs connections for a neuron
        unsigned numOutputs = (l == topology.size() - 1) ? 0 : topology[l + 1];

        // Creating neurons for each layers (plus one bias)
        for(unsigned n = 0; n <= topology[l]; n++) {
            netLayers_.back().push_back(Neuron(numOutputs, n));
        }
    }
}


void Network::feedForward(const std::vector<double> &inputVals) {
    assert(inputVals.size() == netLayers_[0].size() - 1);

    for(unsigned i = 0; i < inputVals.size(); i++) {
        netLayers_[0][i].setOutputVal(inputVals[i]);
    }

    for(unsigned l = 1; l < netLayers_.size(); l++) {
        Layer &prevLayer = netLayers_[l - 1];
        for(unsigned n; n < netLayers_[l].size() - 1; n++) {
            netLayers_[l][n].feedFoward(prevLayer);
        }
    }
}



void Network::backProp(const std::vector<double> &targetVals) {
    Layer &outputLayer = netLayers_.back();
    error = 0.0;

    for (unsigned n = 0; n < outputLayer.size(); ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        error += delta * delta;
    }

    error /= outputLayer.size() - 1;
    error = sqrt(error);


    for(unsigned n = 0; n < outputLayer.size() - 1; n++) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for(unsigned l = netLayers_.size() - 2; l > 0; l--) {
        Layer &hiddenLayer = netLayers_[l];
        Layer &nextLayer = netLayers_[l + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); n++) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    for (unsigned l = netLayers_.size() - 1; l > 0 ; l--) {
        Layer &layer = netLayers_[l];
        Layer &prevLayer = netLayers_[l - 1];

        for(unsigned n = 0; n < layer.size(); n++) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}


void Network::getResults(std::vector<double> &resultVals) const {
    resultVals.clear();

    for(unsigned n = 0; n < netLayers_.back().size() - 1; n++) {
        resultVals.push_back(netLayers_.back()[n].getOutputVal());
    }
}

