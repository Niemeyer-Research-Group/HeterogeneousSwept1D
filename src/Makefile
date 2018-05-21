
include config.mk

CODES		:=oneD #twoD
CODE_DIR	:=$(addprefix $(SOURCEPATH)/, $(CODES))
BIN_DIR		:=$(SOURCEPATH)/bin
UTIL_DIR	:=$(SOURCEPATH)/utilities
UTIL_BIN	:=$(BIN_DIR)/util
EQ_BIN		:=$(BIN_DIR)/eqobj
BINS		:=$(UTIL_BIN) $(EQ_BIN)
IFLAGS		:=$(addprefix -I, $(IPATH))
IFLAGS		+= -I$(CC_ICUDAPATH) -I$(CC_IMPIPATH)

DEF=

UTILC		:= $(wildcard $(UTIL_DIR)/*.c*)
TARG		:= $(notdir $(UTILC))
COBJ		:= $(addsuffix .o, $(basename $(TARG)))

.PHONY: swepts dry

.EXPORT_ALL_VARIABLES:

default: $(BINS) $(addprefix $(UTIL_BIN)/, $(COBJ)) swepts 

$(BINS):
	mkdir -p $(BINS)

dry:
	@echo $(COBJ)

$(UTIL_BIN)/%.o: $(UTIL_DIR)/%.cpp
	g++ -c $< -o $@ $(IFLAGS) $(CFLAGS)

swepts:
	$(MAKE) -C $(CODE_DIR)

clean:
	rm -rf $(BIN_DIR)/*


