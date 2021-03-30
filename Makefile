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


# The default targets to run
#
all: test

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