//go:build integration

package llama

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
)

const serverImage = "serhiiherasymovbndigital/llama-embedserver:latest"

// integrationURL is set once by TestMain and shared across all integration tests.
var integrationURL string

func TestMain(m *testing.M) {
	id, url, err := startContainer()
	if err != nil {
		fmt.Fprintf(os.Stderr, "start container: %v\n", err)
		os.Exit(1)
	}
	integrationURL = url

	code := m.Run()

	stopContainer(id)
	os.Exit(code)
}

func startContainer() (id, url string, err error) {
	out, err := exec.Command("docker", "run", "-d",
		"-p", "127.0.0.1::8080",
		serverImage,
	).Output()
	if err != nil {
		return "", "", fmt.Errorf("docker run: %w", err)
	}
	id = strings.TrimSpace(string(out))

	// Resolve the host port Docker assigned to container port 8080.
	portOut, err := exec.Command("docker", "port", id, "8080").Output()
	if err != nil {
		stopContainer(id)
		return "", "", fmt.Errorf("docker port: %w", err)
	}
	// Output is "<host>:<port>\n", e.g. "127.0.0.1:54321".
	addr := strings.TrimSpace(string(portOut))
	parts := strings.SplitN(addr, ":", 2)
	url = fmt.Sprintf("http://127.0.0.1:%s", parts[len(parts)-1])

	if err := waitForHealth(url, 3*time.Minute); err != nil {
		stopContainer(id)
		return "", "", err
	}
	return id, url, nil
}

func stopContainer(id string) {
	exec.Command("docker", "rm", "-f", id).Run() //nolint:errcheck
}

func waitForHealth(serverURL string, timeout time.Duration) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(serverURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("server at %s not healthy within %s", serverURL, timeout)
}

// plugin returns a Plugin wired to the shared test container.
func plugin(t *testing.T) *Plugin {
	t.Helper()
	return newTestPlugin(integrationURL, defaultHTTPClient())
}

// ── tests ─────────────────────────────────────────────────────────────────

func TestIntegration_SingleEmbed(t *testing.T) {
	resp, err := plugin(t).embed(context.Background(), &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("the quick brown fox", nil)},
	})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("want 1 embedding, got %d", len(resp.Embeddings))
	}
	if len(resp.Embeddings[0].Embedding) == 0 {
		t.Error("embedding vector is empty")
	}
}

func TestIntegration_BatchEmbed(t *testing.T) {
	texts := []string{"first sentence", "second sentence", "third sentence"}
	docs := make([]*ai.Document, len(texts))
	for i, s := range texts {
		docs[i] = ai.DocumentFromText(s, nil)
	}

	resp, err := plugin(t).embed(context.Background(), &ai.EmbedRequest{Input: docs})
	if err != nil {
		t.Fatalf("embed batch: %v", err)
	}
	if len(resp.Embeddings) != len(texts) {
		t.Fatalf("want %d embeddings, got %d", len(texts), len(resp.Embeddings))
	}
	dims := len(resp.Embeddings[0].Embedding)
	for i, emb := range resp.Embeddings {
		if len(emb.Embedding) != dims {
			t.Errorf("embedding[%d]: want %d dims, got %d", i, dims, len(emb.Embedding))
		}
	}
}

// TestIntegration_SingleAndBatchAgree verifies that single-doc and batch-doc
// paths produce identical vectors for the same text.
func TestIntegration_SingleAndBatchAgree(t *testing.T) {
	p := plugin(t)
	text := "hello world"
	ctx := context.Background()

	single, err := p.embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText(text, nil)},
	})
	if err != nil {
		t.Fatalf("single embed: %v", err)
	}

	batch, err := p.embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText(text, nil),
			ai.DocumentFromText("padding so batch path is taken", nil),
		},
	})
	if err != nil {
		t.Fatalf("batch embed: %v", err)
	}

	sv := single.Embeddings[0].Embedding
	bv := batch.Embeddings[0].Embedding
	if len(sv) != len(bv) {
		t.Fatalf("dimension mismatch: single=%d batch=%d", len(sv), len(bv))
	}
	for i := range sv {
		if sv[i] != bv[i] {
			t.Errorf("dim[%d]: single=%f batch=%f", i, sv[i], bv[i])
		}
	}
}

// TestIntegration_DifferentTextsProduceDifferentVectors ensures the model is
// not returning a constant or degenerate embedding.
func TestIntegration_DifferentTextsProduceDifferentVectors(t *testing.T) {
	resp, err := plugin(t).embed(context.Background(), &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("cat", nil),
			ai.DocumentFromText("rocket ship", nil),
		},
	})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}

	a := resp.Embeddings[0].Embedding
	b := resp.Embeddings[1].Embedding
	if len(a) != len(b) {
		t.Fatalf("dimension mismatch: %d vs %d", len(a), len(b))
	}

	var diff float64
	for i := range a {
		d := float64(a[i] - b[i])
		diff += d * d
	}
	if math.Sqrt(diff) < 1e-6 {
		t.Error("different texts produced identical vectors — model may be broken")
	}
}

// TestIntegration_Deterministic verifies that the same input always produces
// the same vector across two calls.
func TestIntegration_Deterministic(t *testing.T) {
	p := plugin(t)
	text := "determinism check"
	ctx := context.Background()

	first, err := p.embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText(text, nil)},
	})
	if err != nil {
		t.Fatalf("first embed: %v", err)
	}
	second, err := p.embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText(text, nil)},
	})
	if err != nil {
		t.Fatalf("second embed: %v", err)
	}

	a := first.Embeddings[0].Embedding
	b := second.Embeddings[0].Embedding
	if len(a) != len(b) {
		t.Fatalf("dimension mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("dim[%d] not deterministic: %f vs %f", i, a[i], b[i])
		}
	}
}

func TestIntegration_EmptyInput(t *testing.T) {
	resp, err := plugin(t).embed(context.Background(), &ai.EmbedRequest{Input: nil})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Embeddings) != 0 {
		t.Errorf("want 0 embeddings, got %d", len(resp.Embeddings))
	}
}
