EXAMPLE_BIN ?= bin/example

.PHONY: help build test test-race test-integration vet tidy clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  %-18s %s\n", $$1, $$2}'

build: ## Verify the library and example compile
	go build ./...

$(EXAMPLE_BIN): ## Build the example binary
	go build -o $(EXAMPLE_BIN) ./example/

test: ## Run unit tests
	go test -v -count=1 ./llama/

test-race: ## Run unit tests with the race detector
	go test -race -v -count=1 ./llama/

test-integration: ## Pull the server image and run integration tests (requires Docker)
	docker pull ghcr.io/content-control-center/llama-embedserver:latest
	go test -v -count=1 -tags integration -timeout 10m ./llama/

vet: ## Run go vet
	go vet ./...

tidy: ## Tidy and verify go.mod / go.sum
	go mod tidy
	go mod verify

clean: ## Remove build artifacts
	rm -rf bin/

.DEFAULT_GOAL := help
