# DEF adds extra defines. Specificially -DNOS. This prevents the program from writing the solution out so it can be 
# performance tested more efficiently.
# run make DEF="-DNOS"


MAKEPATH:=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# LPATH+=

BINPATH:=$(MAKEPATH)bin
EQPATH:=$(MAKEPATH)equations
DECOMPPATH:=$(MAKEPATH)decomposition
UTILPATH:=$(MAKEPATH)utilities

IPATH+=-I$(EQPATH) -I$(DECOMPPATH) -I$(UTILPATH)

DH := $(wildcard $(DECOMPPATH)/*.h) $(EQPATH)/heads.h

CPREQ := $(MAKEPATH)sMain.cu $(UTILPATH)/jsoncpp.cpp

EQINIT:=$(wildcard $(EQPATH)/*.h)

EQS:=$(filter-out $(DH), $(EQINIT))

EQI:=$(foreach eq, $(EQS), $(basename $(notdir $(eq))))

OBJS:=$(foreach eq, $(EQI), $(BINPATH)/$(eq))

XL:=$(foreach lp, $(LPATH), -Xlinker=-rpath,$(lp))

LPATHS:=$(foreach lp, $(LPATH), -L$(lp))

default: $(OBJS) 

$(OBJS): $(BINPATH)%: $(EQPATH)%.h $(CPREQ) $(DH) $(UTILPATH)/gpuDetector.h Makefile
	$(eval IFL:=$(shell echo $(@F) | tr '[:lower:]' '[:upper:]'))
	$(NVCC) $(CPREQ) -o $@ $(IPATH) -D$(IFL) $(DEF) $(CFLAGS) $(CUDAFLAGS) $(XL) $(LPATHS) $(LIBS) 

