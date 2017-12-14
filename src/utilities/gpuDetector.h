
#ifndef DETECT
#define DETECT

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>
#include <sstream>
#include <string>

namespace detection
{

#define RLEN 80

using namespace std;

struct hname{
    int ng;
    char hostname[RLEN];
};

typedef vector<hname> hvec;

int getHost(hvec &ids, hname *newHost)
{
    char machineName[RLEN];
    int rb = RLEN;
    int nGpu;
    cudaGetDeviceCount(&nGpu);
    MPI_Get_processor_name(&machineName[0],  &rb);
    for (int i=0; i<ids.size(); i++)
    {
        if (!strcmp(ids[i].hostname, machineName))
        {
            return i;
        }
    }

    strcpy(newHost->hostname, machineName);
    newHost->ng = nGpu;
    return ids.size();
}

// Test device sight.
bool detector(int ranks, int sz)
{
    hvec ledger;
    int machineID;
    int hasgpu = 0;

	hname hBuf;
    MPI_Datatype atype;
    MPI_Datatype typs[] = {MPI_INT, MPI_CHAR};
    int nm[] = {1, RLEN};
    MPI_Aint disp[] = {0, 4};
    MPI_Type_create_struct(2, nm, disp, typs, &atype);
    MPI_Type_commit(&atype);

    for (int k=0; k<sz; k++)
    {
        if(ranks == k)
        {
            machineID = getHost(ledger, &hBuf);
        }

        MPI_Bcast(&hBuf, 1, atype, k, MPI_COMM_WORLD);
		if (ledger.size() > 0)
		{
			if (strcmp(hBuf.hostname, ledger.back().hostname))
			{
				ledger.push_back(hBuf);
			}
		}
		else
		{
			ledger.push_back(hBuf);
		}
    }

	MPI_Comm machineComm;
    MPI_Comm_split(MPI_COMM_WORLD, machineID, ranks, &machineComm);
    int machineRank, machineSize;
    MPI_Comm_rank(machineComm, &machineRank);
    MPI_Comm_size(machineComm, &machineSize);

    MPI_Barrier(MPI_COMM_WORLD);

	int nGpu = ledger[machineID].ng;
    int pcivec[nGpu*3];
	cudaDeviceProp props;

	MPI_Barrier(MPI_COMM_WORLD);

	if (machineRank == 0)
    {
        for (int k = 0; k < nGpu; k++)
        {
            cudaGetDeviceProperties(&props, k);
            cout << "----------------------" << endl;
			cout << "Rank " << ranks << " device - " << k << " ";
            cout << props.name << " " << props.pciBusID << endl;
            cout << std::hex << props.pciDomainID << ":" <<  props.pciBusID << ":" << props.pciDeviceID << endl;

			pcivec[3*k] = props.pciDomainID;
			pcivec[3*k+1] = props.pciBusID;
			pcivec[3*k+2] = props.pciDeviceID;
    	}
    }

	MPI_Bcast(&pcivec[0], 3*nGpu, MPI_INT, 0, machineComm);
	MPI_Barrier(MPI_COMM_WORLD);

	int nset = 0;
    int dev;
	string pcistr;
	stringstream bufs;

    for (int i = 1; i<machineSize; i++)
    {
        if ((nGpu - nset) == 0)
        {
            break;
        }
        if (i == machineRank)
        {
            bufs << std::hex << pcivec[3*nset] << ":" <<  pcivec[3*nset+1] << ":" <<  pcivec[3*nset+2];

            cudaDeviceGetByPCIBusId(&dev, bufs.str().c_str());
            cudaGetDeviceProperties(&props, dev);


            cout << "Global Rank: " << ranks << " Machine Rank: " << machineRank << std::endl;
            cout << "on machine " << ledger[machineID].hostname << std::endl;
            cout << i << " " << bufs.str() << " " << props.name << endl;

            if (props.kernelExecTimeoutEnabled)
            {
                cout << "DEVICE IS DISPLAY, NOT ASSIGNED" << endl;
            }
            else
            {
                cudaSetDevice(dev);
                cout << "Acquired GPU: " << props.name << " with pciID: " << bufs.str() << endl;
                hasgpu = 1;
            }
            cout << "----------------------" << endl;
            nset++;
        }
        MPI_Bcast(&nset, 1, MPI_INT, i, machineComm);
        MPI_Barrier(machineComm);
    }

    MPI_Type_free(&atype);
    MPI_Comm_free(&machineComm);
    MPI_Barrier(MPI_COMM_WORLD);
    return hasgpu;
}
}
#endif
