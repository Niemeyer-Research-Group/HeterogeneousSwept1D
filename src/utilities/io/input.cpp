#include "input.h"
#include <iostream>
#include <cstdlib>

using namespace std;

eCheckIn void (int dv, int tpb, int argc)
{
    if (argc < 8)
	{
        cout << "NOT ENOUGH ARGUMENTS:" << endl;
		cout << "The Program takes 8 inputs, #Divisions, #Threads/block, deltat, finish time, output frequency..." << endl;
        cout << "Algorithm type, Variable Output File, Timing Output File (optional)" << endl;
		exit(-1);
	}

	if ((dv & (tpb-1) != 0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

}
//Some yaml parsing files and a raw input parser.  AND ERROR PARSERS