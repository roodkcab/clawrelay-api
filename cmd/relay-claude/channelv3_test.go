package main

import (
	"net/http/httptest"
	"strings"
	"testing"

	"clawrelay-api/pkg/openai"
)

// nil usage → NO usage chunk: downstream must keep storing NULL ("not
// metered"), never a fake 0 ("free request").
func TestV3EmitCloseNilUsageOmitsChunk(t *testing.T) {
	rec := httptest.NewRecorder()
	v3EmitClose(rec, rec, "c", 1700000000, "haiku", "hello", true, nil) // includeUsage=true
	body := rec.Body.String()
	if strings.Contains(body, `"prompt_tokens"`) {
		t.Errorf("nil usage must not emit a usage chunk (would log as 0, not NULL):\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing finish chunk:\n%s", body)
	}
	if !strings.Contains(body, `data: [DONE]`) {
		t.Errorf("missing [DONE]:\n%s", body)
	}
}

// Harvested usage + includeUsage → a usage chunk between finish and [DONE].
func TestV3EmitCloseEmitsHarvestedUsage(t *testing.T) {
	rec := httptest.NewRecorder()
	u := openai.BuildUsageInfo(10, 5, 3, 2) // prompt=10+3+2=15, completion=5
	v3EmitClose(rec, rec, "c", 1700000000, "haiku", "hello", true, u)
	body := rec.Body.String()
	if !strings.Contains(body, `"prompt_tokens":15`) || !strings.Contains(body, `"completion_tokens":5`) {
		t.Errorf("missing usage chunk fields:\n%s", body)
	}
	if !strings.Contains(body, `"cached_tokens":3`) || !strings.Contains(body, `"cache_creation_tokens":2`) {
		t.Errorf("missing prompt_tokens_details:\n%s", body)
	}
	finIdx := strings.Index(body, `"finish_reason":"stop"`)
	useIdx := strings.Index(body, `"prompt_tokens"`)
	doneIdx := strings.Index(body, "data: [DONE]")
	if !(finIdx >= 0 && useIdx > finIdx && doneIdx > useIdx) {
		t.Errorf("usage chunk must sit between finish chunk and [DONE] (fin=%d use=%d done=%d):\n%s",
			finIdx, useIdx, doneIdx, body)
	}
}

// includeUsage=false suppresses the chunk even when usage was harvested.
func TestV3EmitCloseRespectsIncludeUsage(t *testing.T) {
	rec := httptest.NewRecorder()
	v3EmitClose(rec, rec, "c", 1700000000, "haiku", "hello", false, openai.BuildUsageInfo(1, 1, 0, 0))
	if body := rec.Body.String(); strings.Contains(body, `"prompt_tokens"`) {
		t.Errorf("includeUsage=false must not emit usage chunk:\n%s", body)
	}
}
