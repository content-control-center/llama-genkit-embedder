package llama

import "net/http"

// Config is passed when constructing the Plugin.
type Config struct {
	// LlamaEmbedServerAddress is the base URL of the llama-embedserver,
	// e.g. "http://localhost:8080". Required.
	LlamaEmbedServerAddress string

	// HTTPClient allows injecting a custom *http.Client (timeouts, TLS, etc.).
	// If nil, a default client with a 30-second timeout is used.
	HTTPClient *http.Client
}

// EmbedOptions is the per-request options type.
// Pass it via ai.WithConfig(&llama.EmbedOptions{...}) on an Embed call.
// Currently unused; defined so future per-request flags don't change call sites.
type EmbedOptions struct{}
