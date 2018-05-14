


#define _USE_MATH_DEFINES
#include <iostream>
#include <ostream>
#include <istream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "../../utilities/matplotlib-cpp/matplotlibcpp.h"
#include <map>

namespace plt = matplotlibcpp;

struct PassIndex
{
    //Edges to pass;
    typedef idxvec std::array<std::vector<int>, 4>;
    const int side, base;
    int nPass, sizePass;
    idxvec pyramid, bridge;
    int *dPyramid, *dBridge

    PassIndex(int side, int base) : side(side), base(base)
    {
        nPass       = side/4 * (side + 2) + (side + 2);
        sizePass    = 4 * nPass * sizeof(int);
        dPyramid    = nullptr;
        dBridge     = nullptr;
    }

    ~PassIndex()
    {
        std::cout << "Hey! Who deleted me! " << std::endl;
    }

    void initialize()
    {
        int ca[4] = {0};
        int flatidx;
        // Pyramid Loop
        for (int ky = 1; ky<=side; ky++)
        {
            for (int kx = 1; kx<=side; kx++)
            {
                flatidx = ky * base + kx;
                if (kx <= ky){
                    if (kx+ky <= side+1) pyramid[0].push_back(flatidx);
                    if (kx+ky >= side+1) pyramid[3].push_back(flatidx);
                }
                if (kx >= ky){
                    if (kx+ky <= side+1) pyramid[1].push_back(flatidx);
                    if (kx+ky >= side+1) pyramid[2].push_back(flatidx);
                }
            }
        }
        // Bridge Loop

        for (int i=0; i<4; i++) ca[i] = 0;
        for (int ky = 0; ky<=side; ky++)
        {
            for (int kx = 0; kx<=side; kx++)
            {
                flatidx = ky * base + kx;
                if (kx <= ky){
                    if (kx+ky <= side) bridge[0].push_back(flatidx);
                    if (kx+ky >= side) bridge[3].push_back(flatidx);
                }
                if (kx >= ky){
                    if (kx+ky <= side) bridge[1].push_back(flatidx);
                    if (kx+ky >= side) bridge[2].push_back(flatidx);
                }
            }
        }
    }

    int *getPtr(const bool INTERIOR)
    {

        if (INTERIOR) return bridge;
        else          return pyramid;
    }

    void plotPass(int eye)
    {

        std::vector<size_t> px, py;
        int neweye, cnew;
        std::string rc[] = {".r", ".b", ".g", ".k"};
        std::string titles[] = {"South", "East", "North", "West"};
        for (int i=0; i<4; i++)
        {
            cnew = 0;
            for (int k=0; k<nPass; k++)
            {
                if (eye){
                    px.push_back(pyramid[i][k] % base);
                    py.push_back(pyramid[i][k] / base);
                }
                else{
                    px.push_back(bridge[i][k] % base);
                    py.push_back(bridge[i][k] / base);
                }
            }

            plt::named_plot( titles[i], px, py,rc[i]);
            std::cout << "Im Plotting!" << px.size() << std::endl;
            neweye = px.back();
            px.pop_back();

            while (neweye == 0){
                std::cout << i << " " << neweye << "   ";
                neweye=px.back();
                px.pop_back();
                cnew++;
            }
            std::cout << "---------" << cnew << std::endl;

            px.clear();
            py.clear();
        }
        plt::title("PyramidPass");
        plt::legend();
        plt::show();
        std::cout << "Im Plotting!" << px.size() << std::endl;
    }
};


int main()
{
    PassIndex pidx(32, 50);
    std::cout << " NumberPassed: " << pidx.nPass << std::endl;
    pidx.initialize();
    pidx.plotPass(0);
    pidx.plotPass(1);
    

    return 0;
}


