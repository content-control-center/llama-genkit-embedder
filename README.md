# llama-genkit-embedder

A [Firebase Genkit](https://github.com/firebase/genkit) Go plugin that exposes the
[llama-embedserver](https://github.com/content-control-center/llama-embedserver) as a
first-class `ai.Embedder`. Drop it into any Genkit application to generate text embeddings
from a locally-running, self-hosted model — no cloud API keys, no data leaving your
infrastructure.

---

## Why this exists

Genkit ships with embedder plugins for hosted services (Google AI, Vertex AI, Ollama with
a remote server). None of them fit a setup where the model is baked into a Docker image,
served over plain HTTP on a local or private network, and operated by your own team.

`llama-embedserver` solves the server side: it packages a GGUF model inside a Docker
image and serves it over HTTP (port 8080) and gRPC (port 9090).

This plugin solves the client side: it wraps those HTTP endpoints in the `api.Plugin` and
`ai.Embedder` interfaces so that Genkit flows, retrievers, and indexers can use the server
without knowing anything about the underlying transport.

---

## Tight coupling with llama-embedserver

This plugin is intentionally coupled to
[`github.com/content-control-center/llama-embedserver`](https://github.com/content-control-center/llama-embedserver).
It is **not** a generic HTTP-to-embedder adapter.

The wire format is fixed to match the server's REST API exactly:

| Plugin call | Server endpoint | Request body | Response body |
|-------------|-----------------|--------------|---------------|
| single document | `POST /embed` | `{"text": "..."}` | `{"embedding": [...], "dimensions": N}` |
| multiple documents | `POST /embed/batch` | `{"texts": ["...", "..."]}` | `{"embeddings": [[...], [...]], "dimensions": N}` |

If you change the server's JSON field names or add authentication, you must update this
plugin to match. The intentional trade-off is simplicity: no schema negotiation, no
version handshake, no feature flags — just two HTTP calls.

---

## Architectural decisions

### HTTP, not gRPC

The server exposes both HTTP (port 8080) and gRPC (port 9090). The plugin uses HTTP for
three reasons:

1. **Zero extra dependencies.** The HTTP client is `net/http` from the standard library.
   gRPC would require `google.golang.org/grpc` and the generated `.pb.go` files.

2. **The proto files are not importable.** The server generates its gRPC code inside
   `package main` (`go_package = ".../cmd/embedserver;main"`), which makes the generated
   types impossible to import from an external module without extracting them into a
   separate proto module first.

3. **No measurable throughput difference.** Embedding calls are already serialised
   server-side behind a mutex (the underlying llama.cpp context is not thread-safe).
   The bottleneck is inference time, not HTTP vs gRPC framing overhead.

A future `grpc.go` could be added alongside `embedder.go` if the proto is ever published
as a separate importable module, without changing the public API.

### Standalone Go module

The plugin lives in its own module rather than inside the `llama-embedserver` repository
for the same reason Genkit's own plugins (Ollama, Vertex, etc.) are separate: consumers
should be able to `go get` the client without pulling in the server's CGO dependencies,
Docker tooling, or llama.cpp build chain.

### Smart single-vs-batch routing

Genkit passes all inputs — even a single document — as `[]*ai.Document`. The plugin
checks `len(req.Input)`:

- **1 document** → `POST /embed`. Avoids the batch wrapper and is marginally simpler on
  the server.
- **N > 1 documents** → `POST /embed/batch`. All texts travel in one HTTP round-trip
  instead of N sequential calls.

This is more efficient than always using the batch endpoint (which has a small JSON
marshalling overhead for a single text) and far more efficient than calling `/embed` N
times in a loop.

### Plugin lifecycle

The plugin follows the pattern of all Genkit first-party plugins:

| Step | What happens |
|------|-------------|
| `llama.New(cfg)` | Creates the plugin; no I/O. |
| `genkit.WithPlugins(plugin)` → `genkit.Init` | Genkit calls `plugin.Init(ctx)`, which validates config and wires up the HTTP client. Panics on misconfiguration — this is a programmer error, not a runtime error. |
| `plugin.DefineEmbedder(g)` | Registers the embedder in Genkit's action registry and returns an `ai.Embedder` handle. |
| `embedder.Embed(ctx, req)` | Called at runtime; dispatches to `/embed` or `/embed/batch`. |

`Init` and `DefineEmbedder` are intentionally separate so that an application can decide
at startup which models to register, without `Init` registering anything implicitly.

### Per-request options

`EmbedOptions` is defined but currently empty. It exists so that callers using
`ai.WithConfig(&llama.EmbedOptions{})` will not need to change their call sites when
future per-request parameters (task type hints, truncation mode) are added.

---

## Installation

```bash
go get github.com/content-control-center/llama-genkit-embedder
```

Requires Go 1.24+ and `github.com/firebase/genkit/go` v1.6+.

---

## Usage

### Start the embedding server

```bash
docker run --rm -p 8080:8080 serhiiherasymovbndigital/llama-embedserver:latest
```

The model is baked into the image — no volume mounts needed.

### Wire the plugin into your Genkit application

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	llama "github.com/content-control-center/llama-genkit-embedder/llama"
)

func main() {
	ctx := context.Background()

	// 1. Create the plugin.
	plugin := llama.New(llama.Config{
		LlamaEmbedServerAddress: "http://localhost:8080", // required
	})

	// 2. Init Genkit — calls plugin.Init internally.
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	// 3. Register the embedder. Returns an ai.Embedder handle.
	embedder := plugin.DefineEmbedder(g)

	// 4a. Embed a single document — routed to POST /embed.
	resp, err := embedder.Embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("the quick brown fox", nil),
		},
	})
	if err != nil {
		log.Fatalf("embed: %v", err)
	}
	fmt.Printf("dims=%d  vec[0]=%f\n",
		len(resp.Embeddings[0].Embedding),
		resp.Embeddings[0].Embedding[0],
	)

	// 4b. Embed multiple documents — routed to POST /embed/batch (one round-trip).
	resp, err = embedder.Embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("the quick brown fox", nil),
			ai.DocumentFromText("jumped over the lazy dog", nil),
			ai.DocumentFromText("pack my box with five dozen liquor jugs", nil),
		},
	})
	if err != nil {
		log.Fatalf("embed batch: %v", err)
	}
	for i, emb := range resp.Embeddings {
		fmt.Printf("[%d] dims=%d\n", i, len(emb.Embedding))
	}
}
```

### Use with a Genkit retriever (RAG)

The embedder returned by `DefineEmbedder` satisfies `ai.Embedder` and can be passed
directly to any Genkit retriever or indexer, for example the built-in local vector store:

```go
import "github.com/firebase/genkit/go/plugins/localvec"

ds, retriever, err := localvec.DefineRetriever(g, "my-docs",
	localvec.Config{Embedder: embedder},
	nil,
)
if err != nil {
	log.Fatal(err)
}

// Index documents
err = ai.Index(ctx, ds,
	ai.WithDocs(ai.DocumentFromText("Go is an open source language", nil)),
)

// Retrieve
results, err := ai.Retrieve(ctx, retriever,
	ai.WithTextDocs("what language is Google famous for?"),
)
```

### Custom HTTP client

Inject a custom `*http.Client` to control timeouts, add auth headers, or use mutual TLS:

```go
plugin := llama.New(llama.Config{
	LlamaEmbedServerAddress: "https://embed.internal",
	ModelName:     "embeddinggemma-300m",
	Dimensions:    896,
	HTTPClient: &http.Client{
		Timeout: 10 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				Certificates: []tls.Certificate{clientCert},
			},
		},
	},
})
```

---

## Repository layout

```
llama-genkit-embedder/
├── go.mod
├── llama/
│   ├── options.go        — Config and EmbedOptions types
│   ├── llama.go          — Plugin struct, Init, DefineEmbedder, IsDefinedEmbedder
│   ├── embedder.go       — HTTP wire types, routing logic, conversion helpers
│   └── embedder_test.go  — unit tests (httptest, no real server required)
└── example/
    └── main.go           — runnable end-to-end example
```
