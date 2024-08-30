#
# Go parameters
#
GOCMD = go
GOBUILD = $(GOCMD) build
GOCLEAN = $(GOCMD) clean
GOTEST = $(GOCMD) test -count=1
GOGET = $(GOCMD) get
GORUN = $(GOCMD) run

# The common parameters
BINARY_NAME = goneat
OUT_DIR = out
LOG_LEVEL = -1

# The default parameters to run the experiment
DATA_DIR=./data
TRIALS_NUMBER=10
LOG_LEVEL=info


# The default targets to run
#
all: test

# The target to run double-pole non Markov experiment
#
run-cartpole-two-non-markov:
	$(GORUN) executor.go -out $(OUT_DIR)/pole2_non-markov \
						 -context $(DATA_DIR)/pole2_non-markov.neat \
						 -genome $(DATA_DIR)/pole2_non-markov_startgenes \
						 -experiment cart_2pole_non-markov \
						 -trials $(TRIALS_NUMBER) \
						 -log_level $(LOG_LEVEL)

# The target to run double-pole Markov experiment
#
run-cartpole-two-markov:
	$(GORUN) executor.go -out $(OUT_DIR)/pole2_markov \
						 -context $(DATA_DIR)/pole2_markov.neat \
						 -genome $(DATA_DIR)/pole2_markov_startgenes \
						 -experiment cart_2pole_markov \
						 -trials $(TRIALS_NUMBER) \
						 -log_level $(LOG_LEVEL)


# The target to run double-pole Markov experiment in parallel objective
# function evaluation mode
#
run-cartpole-two-parallel-markov:
	$(GORUN) executor.go -out $(OUT_DIR)/pole2_markov_parallel \
						 -context $(DATA_DIR)/pole2_markov.neat \
						 -genome $(DATA_DIR)/pole2_markov_startgenes \
						 -experiment cart_2pole_markov_parallel \
						 -trials $(TRIALS_NUMBER) \
						 -log_level $(LOG_LEVEL)

# The target to run single-pole experiment
#
run-cartpole:
	$(GORUN) executor.go -out $(OUT_DIR)/pole1 \
						 -context $(DATA_DIR)/pole1_150.neat \
						 -genome $(DATA_DIR)/pole1startgenes \
						 -experiment cart_pole \
						 -trials $(TRIALS_NUMBER) \
						 -log_level $(LOG_LEVEL)

# The target to run single-pole experiment in parallel objective
# function evaluation mode
#
run-cartpole-parallel:
	$(GORUN) executor.go -out $(OUT_DIR)/pole1_parallel \
						 -context $(DATA_DIR)/pole1_150.neat \
						 -genome $(DATA_DIR)/pole1startgenes \
						 -experiment cart_pole_parallel \
						 -trials 100 \
						 -log_level $(LOG_LEVEL)

# The target to run disconnected XOR experiment
#
run-xor-disconnected:
	$(GORUN) executor.go -out $(OUT_DIR)/xor_disconnected \
						 -context $(DATA_DIR)/xor.neat \
						 -genome $(DATA_DIR)/xordisconnectedstartgenes \
						 -experiment XOR \
						 -trials $(TRIALS_NUMBER) \
						 -log_level $(LOG_LEVEL)

# The target to run XOR experiment
#
run-xor:
	$(GORUN) executor.go -out $(OUT_DIR)/xor \
						 -context $(DATA_DIR)/xor.neat \
						 -genome $(DATA_DIR)/xorstartgenes \
						 -experiment XOR \
						 -trials $(TRIALS_NUMBER) \
						 -log_level $(LOG_LEVEL)

# Run unit tests in short mode
#
test-short:
	$(GOTEST) -v --short ./...

# Run all unit tests
#
test:
	$(GOTEST) -v ./...

# Builds binary
#
build: | $(OUT_DIR)
	$(GOBUILD) -o $(OUT_DIR)/$(BINARY_NAME) -v

# Creates the output directory for build artefacts
#
$(OUT_DIR):
	mkdir -p $@

#
# Clean build targets
#
clean:
	$(GOCLEAN)
	rm -f $(OUT_DIR)/$(BINARY_NAME)
	rm -f $(OUT_DIR)/$(BINARY_UNIX)