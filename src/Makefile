

CODE_DIR:= oneD # twoD
BIN_DIR:= bin
UTIL_DIR:= utilities
UTIL_BIN:= $(BIN_DIR)/util

UTILC:= $(wildcard $(UTIL_DIR)/*.c*)
TARG= $(notdir $(UTILC))
TNAME = $(basename $(TARG))

all: pri $(TNAME) swepts $(APPO)

pri:
	@echo $(TNAME) $(TARG)

%: $(UTIL_DIR)/%.cpp Makefile
	g++ -c $< -o $(UTIL_BIN)/$@.o -O3 -lcudart -lcuda

swepts:
	$(MAKE) -C $(CODE_DIR)

$(APPO)

clean:
	rm $(BIN_DIR)/*.o 

# JSONLIB:=$(BIN_DIR)/jsoncpp.o 
# GPULIB:=$(BIN_DIR)/gpuDetector.o
# utilities/gpuDetector.cu utilities/jsoncpp.cpp

# .PHONY: swepts

# default: $(JSONLIB) $(GPULIB)

# $(JSONLIB): $(UTIL_DIR)/jsoncpp.cpp 
# 	g++ -c $< -o $@ -O3

# $(GPULIB): $(UTIL_DIR)/gpuDetector.cu 
# 	nvcc -c $< -o $@ -std=c++11 -lmpi -O3

