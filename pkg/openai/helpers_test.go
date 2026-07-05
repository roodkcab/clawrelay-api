package openai

import (
	"strings"
	"testing"
)

func TestSanitizeEnvVarsInLogSimpleObject(t *testing.T) {
	in := `{"model":"x","env_vars":{"API_KEY":"secret1","TOKEN":"secret2"},"stream":true}`
	got := SanitizeEnvVarsInLog(in)
	want := `{"model":"x","env_vars":{"[REDACTED]":"..."},"stream":true}`
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// TestSanitizeEnvVarsInLogValueWithBrace locks in the fix: the old
// `[^}]*` regex stopped at the '}' inside the first value, leaking every
// key/value after it into the log verbatim.
func TestSanitizeEnvVarsInLogValueWithBrace(t *testing.T) {
	in := `{"env_vars":{"TEMPLATE":"{a}","API_KEY":"leaked-secret"},"model":"x"}`
	got := SanitizeEnvVarsInLog(in)
	if strings.Contains(got, "leaked-secret") {
		t.Fatalf("secret after a brace-containing value leaked: %q", got)
	}
	want := `{"env_vars":{"[REDACTED]":"..."},"model":"x"}`
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// TestSanitizeEnvVarsInLogValueWithEscapedQuote: `\"` inside a value must not
// end the string early (a following `}` in the value would then close the
// scan prematurely).
func TestSanitizeEnvVarsInLogValueWithEscapedQuote(t *testing.T) {
	in := `{"env_vars":{"MSG":"say \"}\" loudly","API_KEY":"leaked-secret"},"model":"x"}`
	got := SanitizeEnvVarsInLog(in)
	if strings.Contains(got, "leaked-secret") {
		t.Fatalf("secret after an escaped-quote value leaked: %q", got)
	}
	want := `{"env_vars":{"[REDACTED]":"..."},"model":"x"}`
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// TestSanitizeEnvVarsInLogValueWithBackslash: a value ending in `\\` means
// the next '"' really closes the string — the scanner must not treat the
// second backslash as escaping the quote and overrun.
func TestSanitizeEnvVarsInLogValueWithBackslash(t *testing.T) {
	in := `{"env_vars":{"WIN_PATH":"C:\\dir\\"},"model":"x"}`
	got := SanitizeEnvVarsInLog(in)
	want := `{"env_vars":{"[REDACTED]":"..."},"model":"x"}`
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestSanitizeEnvVarsInLogMultipleOccurrences(t *testing.T) {
	in := `{"a":{"env_vars":{"K":"v}1"}},"b":{"env_vars":{"K2":"v2"}}}`
	got := SanitizeEnvVarsInLog(in)
	want := `{"a":{"env_vars":{"[REDACTED]":"..."}},"b":{"env_vars":{"[REDACTED]":"..."}}}`
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestSanitizeEnvVarsInLogNoEnvVars(t *testing.T) {
	in := `{"model":"x","messages":[{"role":"user","content":"hi"}]}`
	if got := SanitizeEnvVarsInLog(in); got != in {
		t.Fatalf("body without env_vars was altered: %q", got)
	}
}

// TestSanitizeEnvVarsInLogTruncatedObject: an unterminated env_vars object
// (body cut mid-value) must redact through to the end rather than leak.
func TestSanitizeEnvVarsInLogTruncatedObject(t *testing.T) {
	in := `{"env_vars":{"API_KEY":"leaked-secr`
	got := SanitizeEnvVarsInLog(in)
	if strings.Contains(got, "leaked-secr") {
		t.Fatalf("truncated object leaked its content: %q", got)
	}
}

// TestSanitizeEnvVarsInLogNonObjectValue: `"env_vars"` followed by something
// other than an object (e.g. an already-redacted string) passes through.
func TestSanitizeEnvVarsInLogNonObjectValue(t *testing.T) {
	in := `{"env_vars":"[REDACTED]","model":"x"}`
	if got := SanitizeEnvVarsInLog(in); got != in {
		t.Fatalf("non-object env_vars value was altered: %q", got)
	}
}
