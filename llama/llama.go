// Package llama provides a Genkit plugin for the llama-go-docker embedding server.
//
// Usage:
//
//	plugin := llama.New(llama.Config{
//	    LlamaEmbedServerAddress: "http://localhost:8080",
//	    ModelName:     "embeddinggemma-300m",
//	    Dimensions:    896,
//	})
//
//	g, err := genkit.Init(ctx, genkit.WithPlugins(plugin))
//	if err != nil { ... }
//
//	embedder := plugin.DefineEmbedder(g)
//
//	resp, err := ai.Embed(ctx, embedder, ai.WithTextDocs("the quick brown fox"))
package llama

import (
	"context"
	"fmt"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "llama"

// Plugin implements api.Plugin for the llama-go-docker embedding server.
// Create one with New and pass it to genkit.WithPlugins.
type Plugin struct {
	cfg     Config
	mu      sync.Mutex
	initted bool
}

// New creates a Plugin. Pass it to genkit.WithPlugins(plugin) so that Genkit
// calls Init during startup. Then call DefineEmbedder to register the embedder.
func New(cfg Config) *Plugin {
	return &Plugin{cfg: cfg}
}

// Name satisfies api.Plugin — used as the provider prefix in action names.
func (p *Plugin) Name() string { return provider }

// Init satisfies api.Plugin. Called once by the Genkit runtime.
// Panics on misconfiguration, following the convention of all first-party plugins.
func (p *Plugin) Init(_ context.Context) []api.Action {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.initted {
		panic("llama.Plugin.Init: already called")
	}
	if p.cfg.LlamaEmbedServerAddress == "" {
		panic("llama.Plugin.Init: Config.LlamaEmbedServerAddress is required")
	}
	if p.cfg.HTTPClient == nil {
		p.cfg.HTTPClient = defaultHTTPClient()
	}
	p.initted = true
	return nil
}

// DefineEmbedder contacts the server's GET /info endpoint to discover the
// model's embedding dimensions, registers the embedder with the given Genkit
// instance, and returns a handle to it. Call this after genkit.Init.
//
// The embedder is registered under the name "llama/<LlamaEmbedServerAddress>".
// For 1-document requests it calls POST /embed; for N>1 it calls POST /embed/batch.
func (p *Plugin) DefineEmbedder(g *genkit.Genkit) (ai.Embedder, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.initted {
		panic("llama.Plugin.DefineEmbedder: Init has not been called — pass the plugin to genkit.WithPlugins")
	}
	info, err := p.fetchInfo(context.Background())
	if err != nil {
		return nil, fmt.Errorf("llama: fetch server info: %w", err)
	}
	name := fmt.Sprintf("%s/%s", provider, p.cfg.LlamaEmbedServerAddress)
	opts := &ai.EmbedderOptions{
		Label:      "Llama @ " + p.cfg.LlamaEmbedServerAddress,
		Dimensions: info.Dimensions,
		Supports: &ai.EmbedderSupports{
			Input: []string{"text"},
		},
	}
	return genkit.DefineEmbedder(g, name, opts,
		func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
			return p.embed(ctx, req)
		},
	), nil
}

// IsDefinedEmbedder reports whether an embedder for the given server address
// has already been registered on g.
func IsDefinedEmbedder(g *genkit.Genkit, serverAddr string) bool {
	return genkit.LookupEmbedder(g, fmt.Sprintf("%s/%s", provider, serverAddr)) != nil
}
