# DEF adds extra defines. Specificially -DNOS. This prevents the program from writing the solution out so it can be 
# performance tested more efficiently.
# run make DEF="-DNOS"


MAKEPATH:=$(SOURCEPATH)/oneD

# LPATH+=

BINPATH		:=$(MAKEPATH)/bin
EQPATH		:=$(MAKEPATH)/equations
DECOMPPATH	:=$(MAKEPATH)/decomposition
UTILPATH	:=$(MAKEPATH)/utilities

IFLAGS		+= -I$(EQPATH) -I$(DECOMPPATH) -I$(UTIL_DIR)

XL			:=$(foreach lp, $(LPATH), -Xlinker=-rpath,$(lp))

DH			:=$(wildcard $(DECOMPPATH)/*.h) $(EQPATH)/heads.h

CPREQ		:=$(MAKEPATH)/sMain.cu

EQINIT		:=$(wildcard $(EQPATH)/*.h)

EQS			:=$(filter-out $(DH), $(EQINIT))

EQI			:=$(foreach eq, $(EQS), $(basename $(notdir $(eq))))

EQOBJ		:=$(addsuffix .o, $(EQI))

default: $(addprefix $(EQ_BIN)/, $(EQOBJ)) $(addprefix $(BIN_DIR)/, $(EQI))

$(BIN_DIR)/%: $(EQ_BIN)/%.o $(UTIL_BIN)/*.o
	$(NVCC) $^ -o $@ $(XL) $(LPATHS) $(LIBS) $(LIBMPI) $(CUDAFLAGS) $(CFLAGS)

$(EQ_BIN)/%.o: $(EQPATH)/%.h $(CPREQ) $(DH)
	$(eval IFL:=$(shell echo $(basename $(@F)) | tr '[:lower:]' '[:upper:]'))
	$(NVCC) -c $(CPREQ) -o $@ $(IFLAGS) -D$(IFL) $(DEF) $(CUDAFLAGS) $(CFLAGS)

