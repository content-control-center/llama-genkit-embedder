// Example shows how to wire the llama plugin into a Genkit application.
//
// Prerequisites:
//
//	docker run --rm -p 8080:8080 llama-embedserver
//
// Run:
//
//	go run ./example
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

	plugin := llama.New(llama.Config{
		LlamaEmbedServerAddress: "http://localhost:8080",
	})

	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	embedder, err := plugin.DefineEmbedder(g)
	if err != nil {
		log.Fatalf("define embedder: %v", err)
	}

	// Single document
	resp, err := embedder.Embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("the quick brown fox", nil),
		},
	})
	if err != nil {
		log.Fatalf("embed: %v", err)
	}
	fmt.Printf("single: dims=%d\n", len(resp.Embeddings[0].Embedding))

	// Multiple documents — routed to POST /embed/batch in one HTTP round-trip
	resp, err = embedder.Embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("the quick brown fox", nil),
			ai.DocumentFromText("jumped over the lazy dog", nil),
		},
	})
	if err != nil {
		log.Fatalf("embed batch: %v", err)
	}
	fmt.Printf("batch:  count=%d dims=%d\n", len(resp.Embeddings), len(resp.Embeddings[0].Embedding))
}
