package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestChatCompletionsRejectsOversizedBody: bodies past the 64 MiB cap must be
// refused with 413 instead of being buffered whole into memory.
func TestChatCompletionsRejectsOversizedBody(t *testing.T) {
	body := bytes.NewReader(make([]byte, 64<<20+1))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	rec := httptest.NewRecorder()

	chatCompletionsHandler(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want %d; body: %s", rec.Code, http.StatusRequestEntityTooLarge, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "byte limit") {
		t.Fatalf("expected explicit limit message, got: %s", rec.Body.String())
	}
}

// TestChatCompletionsInvalidJSONStill400: the pre-existing 400 for malformed
// bodies under the limit must not be affected by the size cap.
func TestChatCompletionsInvalidJSONStill400(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{not json"))
	rec := httptest.NewRecorder()

	chatCompletionsHandler(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d; body: %s", rec.Code, http.StatusBadRequest, rec.Body.String())
	}
}
