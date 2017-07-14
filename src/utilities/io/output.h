#ifndef OUTPUTS_H
#define OUTPUTS_H

#include <fstream>
#include <ostream>
#include "json.hpp"

json solution;
json timing;

// Need something to read the dict fields from the equation specific source. 
// Also need something to apply the functions that give the correct output values.

void initializeOutStreams();

void solutionOutput();

void timingOutput();

#endif