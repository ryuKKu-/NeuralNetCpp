//
// Created by Luc on 14/10/2015.
//

#ifndef NEURALNET_NETWORK_H
#define NEURALNET_NETWORK_H

#include <cmath>
#include <vector>
#include "Neuron.h"


class Network {

public:
    Network(const std::vector<double> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;

private:
    std::vector<Layer> netLayers_;
    double error;
};


#endif //NEURALNET_NETWORK_H
