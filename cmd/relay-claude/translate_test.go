package main

import (
	"net/http/httptest"
	"strings"
	"testing"
)

// TestErrorResultSurfacesErrorText: a result with is_error=true and no prior
// streamed delta must emit the error explanation (claude puts it in Result)
// as a content delta before the finish chunk — not end an empty turn.
func TestErrorResultSurfacesErrorText(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newSSETranslator("chatcmpl-err", 1700000000, "haiku", "", identityMeter{}, "")

	line := `{"type":"result","subtype":"error_during_execution","is_error":true,"result":"Something exploded mid-turn"}`
	if got := tr.feed(rec, rec, line, false); got != outcomeContinue {
		t.Fatalf("feed(error result) = %v, want outcomeContinue", got)
	}

	body := rec.Body.String()
	if !strings.Contains(body, "Something exploded mid-turn") {
		t.Fatalf("error text was dropped; SSE output:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("missing finish chunk with finish_reason=stop; SSE output:\n%s", body)
	}
	// The error delta must precede the finish chunk.
	if strings.Index(body, "Something exploded mid-turn") > strings.Index(body, `"finish_reason":"stop"`) {
		t.Fatalf("error text emitted after finish chunk; SSE output:\n%s", body)
	}
	if !tr.sawResult {
		t.Fatal("sawResult not set by error result")
	}
}

// TestErrorResultSubtypeOnly: some CLI versions flag failure only through
// subtype=error_* (no is_error) — the prefix check must still catch it.
func TestErrorResultSubtypeOnly(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newSSETranslator("chatcmpl-err2", 1700000000, "haiku", "", identityMeter{}, "")

	line := `{"type":"result","subtype":"error_max_turns","result":"Reached max turns"}`
	tr.feed(rec, rec, line, false)

	body := rec.Body.String()
	if !strings.Contains(body, "Reached max turns") {
		t.Fatalf("subtype-only error text was dropped; SSE output:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("missing finish chunk; SSE output:\n%s", body)
	}
}

// TestErrorResultAfterStreamedDeltas: once real deltas streamed, the error
// Result text is not re-emitted (the user already saw partial output; the
// tail explanation would duplicate/confuse), but the finish chunk still goes
// out and the error is only logged.
func TestErrorResultAfterStreamedDeltas(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newSSETranslator("chatcmpl-err3", 1700000000, "haiku", "", identityMeter{}, "")

	delta := `{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial answer"}}}`
	tr.feed(rec, rec, delta, false)
	line := `{"type":"result","subtype":"error_during_execution","is_error":true,"result":"boom"}`
	tr.feed(rec, rec, line, false)

	body := rec.Body.String()
	if strings.Contains(body, "boom") {
		t.Fatalf("error text duplicated after streamed deltas; SSE output:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("missing finish chunk; SSE output:\n%s", body)
	}
}

// TestSuccessResultUnchanged: a normal success result must not emit its
// Result text (that is the already-streamed final answer, re-emitting would
// duplicate it) — only the finish chunk.
func TestSuccessResultUnchanged(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newSSETranslator("chatcmpl-ok", 1700000000, "haiku", "", identityMeter{}, "")

	line := `{"type":"result","subtype":"success","result":"final answer text"}`
	tr.feed(rec, rec, line, false)

	body := rec.Body.String()
	if strings.Contains(body, "final answer text") {
		t.Fatalf("success result text should not be emitted; SSE output:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("missing finish chunk; SSE output:\n%s", body)
	}
}
