# DEF adds extra defines. Specificially -DNOS. This prevents the program from writing the solution out so it can be 
# performance tested more efficiently.
# run make DEF="-DNOS"


ODIR:= $(SOURCEPATH)/oneD

EQPATH:=$(ODIR)/equations
DECOMPPATH:=$(ODIR)/decomposition

IFLAGS+= -I$(EQPATH) -I$(DECOMPPATH) -I$(UTIL_DIR)

XL:=$(addprefix -Xlinker=-rpath, $(LPATH))

LPATHS:=$(addprefix -L, $(LPATH))

DH:= $(wildcard $(DECOMPPATH)/*.h) $(EQPATH)/heads.h

CPREQ:=$(ODIR)/sMain.cu 

EQINIT:=$(wildcard $(EQPATH)/*.h)

EQS:=$(filter-out $(DH), $(EQINIT))

EQI:=$(foreach eq, $(EQS), $(basename $(notdir $(eq))))

EQOBJ:=$(addsuffix .o, $(EQI))

default: $(addprefix $(EQ_BIN)/, $(EQOBJ)) $(addprefix $(BIN_DIR)/, $(EQI))

$(BIN_DIR)/%: $(EQ_BIN)/%.o $(UTIL_BIN)/*.o 
	$(NVCC) $^ -o $@ $(XL) $(LPATHS) $(LIBS) $(LIBMPI)

$(EQ_BIN)/%.o: $(EQPATH)/%.h $(CPREQ) $(DH)
	$(eval IFL:=$(shell echo $(basename $(@F)) | tr '[:lower:]' '[:upper:]'))
	$(NVCC) -c $(CPREQ) -o $@ $(IFLAGS) -D$(IFL) $(DEF) $(CFLAGS) $(CUDAFLAGS) 

