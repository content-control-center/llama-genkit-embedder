package llama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
)

// ── wire types matching the server's REST API ─────────────────────────────

type infoResponse struct {
	Dimensions int `json:"dimensions"`
}

type httpEmbedRequest struct {
	Text string `json:"text"`
}

type httpEmbedResponse struct {
	Embedding  []float32 `json:"embedding"`
	Dimensions int       `json:"dimensions"`
}

type httpBatchRequest struct {
	Texts []string `json:"texts"`
}

type httpBatchResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Dimensions int         `json:"dimensions"`
}

// ── core embed logic ──────────────────────────────────────────────────────

func (p *Plugin) embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	if len(req.Input) == 0 {
		return &ai.EmbedResponse{}, nil
	}

	texts := extractTexts(req.Input)

	var embeddings [][]float32
	var err error

	if len(texts) == 1 {
		var vec []float32
		vec, err = p.embedSingle(ctx, texts[0])
		if err != nil {
			return nil, err
		}
		embeddings = [][]float32{vec}
	} else {
		embeddings, err = p.embedBatch(ctx, texts)
		if err != nil {
			return nil, err
		}
	}

	return buildEmbedResponse(embeddings), nil
}

func (p *Plugin) embedSingle(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(httpEmbedRequest{Text: text})
	if err != nil {
		return nil, fmt.Errorf("llama: marshal embed request: %w", err)
	}

	httpResp, err := p.doPost(ctx, "/embed", body)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("llama: /embed returned HTTP %d", httpResp.StatusCode)
	}

	var resp httpEmbedResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("llama: decode /embed response: %w", err)
	}
	return resp.Embedding, nil
}

func (p *Plugin) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(httpBatchRequest{Texts: texts})
	if err != nil {
		return nil, fmt.Errorf("llama: marshal batch request: %w", err)
	}

	httpResp, err := p.doPost(ctx, "/embed/batch", body)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("llama: /embed/batch returned HTTP %d", httpResp.StatusCode)
	}

	var resp httpBatchResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("llama: decode /embed/batch response: %w", err)
	}
	return resp.Embeddings, nil
}

// ── HTTP helpers ──────────────────────────────────────────────────────────

// fetchInfo calls GET /info on the server and returns its metadata.
// The server computes dimensions once at startup via a probe embed, so this
// call is cheap and has no inference cost.
func (p *Plugin) fetchInfo(ctx context.Context) (*infoResponse, error) {
	url := strings.TrimRight(p.cfg.LlamaEmbedServerAddress, "/") + "/info"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("llama: build /info request: %w", err)
	}
	resp, err := p.cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("llama: GET /info: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("llama: /info returned HTTP %d", resp.StatusCode)
	}
	var info infoResponse
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("llama: decode /info response: %w", err)
	}
	return &info, nil
}

func (p *Plugin) doPost(ctx context.Context, path string, body []byte) (*http.Response, error) {
	url := strings.TrimRight(p.cfg.LlamaEmbedServerAddress, "/") + path
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llama: build request for %s: %w", path, err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("llama: POST %s: %w", path, err)
	}
	return resp, nil
}

func defaultHTTPClient() *http.Client {
	return &http.Client{Timeout: 30 * time.Second}
}

// ── conversion helpers ────────────────────────────────────────────────────

// extractTexts concatenates the text Parts of each Document into a single
// string per document, mirroring the concatenateText pattern in the Ollama plugin.
func extractTexts(docs []*ai.Document) []string {
	texts := make([]string, len(docs))
	for i, doc := range docs {
		var sb strings.Builder
		for _, part := range doc.Content {
			sb.WriteString(part.Text)
		}
		texts[i] = sb.String()
	}
	return texts
}

// buildEmbedResponse converts raw float slices into Genkit's ai.EmbedResponse.
func buildEmbedResponse(embeddings [][]float32) *ai.EmbedResponse {
	resp := &ai.EmbedResponse{
		Embeddings: make([]*ai.Embedding, len(embeddings)),
	}
	for i, vec := range embeddings {
		resp.Embeddings[i] = &ai.Embedding{Embedding: vec}
	}
	return resp
}
