#ifndef DETECT
#define DETECT

#define RLEN 80

using namespace std;

struct hname{
    int ng;
    char hostname[RLEN];
};

typedef vector<hname> hvec;

int getHost(hvec &ids, hname *newHost);

int detector(int ranko, const int sz);

#endif