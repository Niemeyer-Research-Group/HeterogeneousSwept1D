#ifndef DETECT
#define DETECT

#define RLEN 80

#include <vector>

struct hname{
    int ng;
    char hostname[RLEN];
};

typedef std::vector<hname> hvec;

int getHost(hvec &ids, hname *newHost);

int detector(const int ranko, const int sz, const int startpos);

#endif