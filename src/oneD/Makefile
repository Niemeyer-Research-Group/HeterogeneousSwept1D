# DEF adds extra defines. Specificially -DNOS. This prevents the program from writing the solution out so it can be 
# performance tested more efficiently.
# run make DEF="-DNOS"


MAKEPATH:=$(SOURCEPATH)/oneD/

# LPATH+=

BINPATH:=$(MAKEPATH)bin
EQPATH:=$(MAKEPATH)equations
DECOMPPATH:=$(MAKEPATH)decomposition
UTILPATH:=$(MAKEPATH)utilities

IFLAGS+= -I$(EQPATH) -I$(DECOMPPATH) -I$(UTIL_DIR)

XL:=$(foreach lp, $(LPATH), -Xlinker=-rpath,$(lp))

LPATHS:=$(addprefix -L, $(LPATH))

DH:= $(wildcard $(DECOMPPATH)/*.h) $(EQPATH)/heads.h

CPREQ:=$(ODIR)/sMain.cu

EQINIT:=$(wildcard $(EQPATH)/*.h)

EQS:=$(filter-out $(DH), $(EQINIT))

EQI:=$(foreach eq, $(EQS), $(basename $(notdir $(eq))))

EQOBJ:=$(addsuffix .o, $(EQI))

.PHONY: ech ia

default: ech heat.o $(BIN_DIR)/heat $(addprefix $(BIN_DIR)/, $(EQI))

ech:
	@echo $(MAKEPATH)
	@echo $(EQ_BIN) $(EQPATH) "\n"$(BIN_DIR) $(EQI)
	@echo $(addprefix $(EQ_BIN)/, $(EQOBJ)) $(addprefix $(BIN_DIR)/, $(EQI))

# ia: $(BIN_DIR)/%: $(EQ_BIN)/%.o
# 	@echo "IA" $@ $^

heat.o:
	$(NVCC) -c sMain.cu -o heat.o -DHEAT $(IFLAGS) $(CFLAGS) $(CUDAFLAGS) $(XL) $(LPATHS) $(LIBS) $(LIBMPI)

$(BIN_DIR)/heat:
	$(NVCC) $(UTIL_BIN)/gpuDetector.o $(UTIL_BIN)/jsoncpp.o $(MAKEPATH)heat.o -o $(BIN_DIR)/heat $(CFLAGS) $(CUDAFLAGS) $(LPATHS) $(LIBS) $(LIBMPI) $(XL)

$(BIN_DIR)/%: $(EQ_BIN)/%.o $(UTIL_BIN)/*.o
	$(NVCC) $^ -o $@ $(XL) $(LPATHS) $(LIBS) $(LIBMPI)

$(EQ_BIN)/%.o: $(EQPATH)/%.h $(CPREQ) $(DH)
	$(eval IFL:=$(shell echo $(basename $(@F)) | tr '[:lower:]' '[:upper:]'))
	$(NVCC) -c $(CPREQ) -o $@ $(IFLAGS) -D$(IFL) $(DEF) $(CFLAGS) $(CUDAFLAGS)

