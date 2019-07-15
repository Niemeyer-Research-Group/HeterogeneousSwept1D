# config.mk 

#COMPILE with -DNOS to not output jsons

include config.mk

CODES		:=oneD #twoD
CODE_DIR	:=$(addprefix $(SOURCEPATH), $(CODES))
BIN_DIR		:=$(SOURCEPATH)bin
UTIL_DIR	:=$(SOURCEPATH)utilities
UTIL_BIN	:=$(BIN_DIR)/util
TESTDIR     :=$(UTIL_BIN)/tests
EQ_BIN		:=$(BIN_DIR)/eqobj
BINS		:=$(UTIL_BIN) $(EQ_BIN) $(TESTDIR)
IFLAGS		:=$(addprefix -I, $(IPATH))
LPATHS		:=$(addprefix -L, $(LPATH))

DEF			=

UTILC		:= $(wildcard $(UTIL_DIR)/*.c*)
TARG		:= $(notdir $(UTILC))
COBJ		:= $(addsuffix .o, $(basename $(TARG)))
TESTSN		:=detectorTest

.PHONY: swepts dry runtests

.EXPORT_ALL_VARIABLES:

default: $(BINS) $(addprefix $(UTIL_BIN)/, $(COBJ)) swepts $(TESTDIR)/$(TESTSN)

$(BINS):
	mkdir -p $(BINS)

dry:
	@echo $(COBJ)

$(UTIL_BIN)/%.o: $(UTIL_DIR)/%.cpp
	$(MPICXX) -c $< -o $@ $(IFLAGS) $(CFLAGS)

swepts: Makefile 
	@mv $(UTIL_BIN)/*Test.o $(TESTDIR)
	$(MAKE) -C $(CODE_DIR)

clean:
	rm -rf $(BIN_DIR)/*

$(TESTDIR)/%Test: $(TESTDIR)/%Test.o swepts
	$(NVCC) $(UTIL_BIN)/gpuDetector.o $< -o $@ $(LPATHS) $(LIBMPI)

runtests: 
	@echo $(TESTSN)
	srun -n 16 $(TESTDIR)/$(TESTSN)
