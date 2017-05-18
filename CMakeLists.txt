cmake_minimum_required(VERSION 3.3)
project(NeuralNet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Network.cpp Network.h Neuron.cpp Neuron.h)
add_executable(NeuralNet ${SOURCE_FILES})