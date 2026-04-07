package llama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// newTestPlugin returns an initialised Plugin pointed at the given test server URL.
func newTestPlugin(serverURL string, client *http.Client) *Plugin {
	return &Plugin{
		cfg: Config{
			LlamaEmbedServerAddress: serverURL,
			HTTPClient:    client,
		},
		initted: true,
	}
}

func TestEmbed_EmptyInput(t *testing.T) {
	p := newTestPlugin("http://unused", http.DefaultClient)
	resp, err := p.embed(context.Background(), &ai.EmbedRequest{Input: nil})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Embeddings) != 0 {
		t.Errorf("want 0 embeddings, got %d", len(resp.Embeddings))
	}
}

func TestEmbed_SingleDocument(t *testing.T) {
	want := []float32{0.1, 0.2, 0.3}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed" {
			t.Errorf("expected /embed, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		var req httpEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Text == "" {
			t.Error("expected non-empty text")
		}
		json.NewEncoder(w).Encode(httpEmbedResponse{Embedding: want, Dimensions: len(want)})
	}))
	defer srv.Close()

	p := newTestPlugin(srv.URL, srv.Client())
	resp, err := p.embed(context.Background(), &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("hello world", nil)},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("want 1 embedding, got %d", len(resp.Embeddings))
	}
	got := resp.Embeddings[0].Embedding
	if len(got) != len(want) {
		t.Fatalf("want %d dims, got %d", len(want), len(got))
	}
	for i, v := range got {
		if v != want[i] {
			t.Errorf("dim[%d]: want %f, got %f", i, want[i], v)
		}
	}
}

func TestEmbed_BatchDocuments(t *testing.T) {
	vecs := [][]float32{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed/batch" {
			t.Errorf("expected /embed/batch, got %s", r.URL.Path)
		}
		var req httpBatchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Texts) != 3 {
			t.Errorf("want 3 texts, got %d", len(req.Texts))
		}
		json.NewEncoder(w).Encode(httpBatchResponse{Embeddings: vecs, Dimensions: 2})
	}))
	defer srv.Close()

	p := newTestPlugin(srv.URL, srv.Client())
	resp, err := p.embed(context.Background(), &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("first", nil),
			ai.DocumentFromText("second", nil),
			ai.DocumentFromText("third", nil),
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Embeddings) != 3 {
		t.Fatalf("want 3 embeddings, got %d", len(resp.Embeddings))
	}
	for i, emb := range resp.Embeddings {
		if len(emb.Embedding) != 2 {
			t.Errorf("embedding[%d]: want 2 dims, got %d", i, len(emb.Embedding))
		}
	}
}

func TestEmbed_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := newTestPlugin(srv.URL, srv.Client())
	_, err := p.embed(context.Background(), &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("hello", nil)},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestEmbed_MultiPartDocument(t *testing.T) {
	var captured string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req httpEmbedRequest
		json.NewDecoder(r.Body).Decode(&req) //nolint:errcheck
		captured = req.Text
		json.NewEncoder(w).Encode(httpEmbedResponse{Embedding: []float32{0.1}, Dimensions: 1})
	}))
	defer srv.Close()

	p := newTestPlugin(srv.URL, srv.Client())
	doc := &ai.Document{
		Content: []*ai.Part{
			{Text: "hello "},
			{Text: "world"},
		},
	}
	_, err := p.embed(context.Background(), &ai.EmbedRequest{Input: []*ai.Document{doc}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if captured != "hello world" {
		t.Errorf("want %q, got %q", "hello world", captured)
	}
}

func TestFetchInfo_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/info" || r.Method != http.MethodGet {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		json.NewEncoder(w).Encode(infoResponse{Dimensions: 896})
	}))
	defer srv.Close()

	p := newTestPlugin(srv.URL, srv.Client())
	info, err := p.fetchInfo(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if info.Dimensions != 896 {
		t.Errorf("want dimensions=896, got %d", info.Dimensions)
	}
}

func TestFetchInfo_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := newTestPlugin(srv.URL, srv.Client())
	_, err := p.fetchInfo(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}
