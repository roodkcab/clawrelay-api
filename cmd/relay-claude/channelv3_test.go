package main

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestV3EmitCloseOmitsUsage(t *testing.T) {
	rec := httptest.NewRecorder()
	v3EmitClose(rec, rec, "c", 1700000000, "haiku", "hello", true) // includeUsage=true
	body := rec.Body.String()
	if strings.Contains(body, `"prompt_tokens"`) {
		t.Errorf("V3 must not emit usage (would log as 0, not NULL):\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing finish chunk:\n%s", body)
	}
	if !strings.Contains(body, `data: [DONE]`) {
		t.Errorf("missing [DONE]:\n%s", body)
	}
}
