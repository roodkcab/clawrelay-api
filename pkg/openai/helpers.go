package openai

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// GenerateChatID returns a fresh `chatcmpl-<hex>` id of the form OpenAI uses.
func GenerateChatID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return "chatcmpl-" + hex.EncodeToString(b)
}

// GenerateToolCallID returns a fresh `call_<hex>` id for tool_calls.
func GenerateToolCallID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return "call_" + hex.EncodeToString(b)
}

// SetCORSHeaders allows the configured origins; falls through silently for
// origins outside the list (browsers will then block based on absent header).
func SetCORSHeaders(w http.ResponseWriter, r *http.Request, allowedOrigins []string) {
	origin := r.Header.Get("Origin")
	for _, allowed := range allowedOrigins {
		if origin == allowed {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			break
		}
	}
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.Header().Set("Access-Control-Max-Age", "3600")
}

// WriteError emits an OpenAI-shaped error envelope.
func WriteError(w http.ResponseWriter, statusCode int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    errType,
			"code":    statusCode,
		},
	})
}

// SanitizeEnvVarsInLog redacts every env_vars object before request bodies
// hit the log file, since the values may contain user-provided secrets. This
// used to be a regex matching `\{[^}]*\}`, but a '}' inside a value ended the
// match early and everything after it — including further secrets — leaked
// into the log verbatim. So the object is now walked with a small string-aware
// scanner that honors escapes and brace depth to find its real closing brace.
func SanitizeEnvVarsInLog(body string) string {
	const key = `"env_vars"`
	var out strings.Builder
	for {
		idx := strings.Index(body, key)
		if idx < 0 {
			break
		}
		// Skip `"env_vars"` then optional whitespace, ':', whitespace, '{'.
		// Anything else (e.g. an already-redacted string value) is copied
		// through untouched and the search resumes after the key.
		j := idx + len(key)
		for j < len(body) && isJSONSpace(body[j]) {
			j++
		}
		if j >= len(body) || body[j] != ':' {
			out.WriteString(body[:j])
			body = body[j:]
			continue
		}
		j++
		for j < len(body) && isJSONSpace(body[j]) {
			j++
		}
		if j >= len(body) || body[j] != '{' {
			out.WriteString(body[:j])
			body = body[j:]
			continue
		}
		end := scanJSONObjectEnd(body, j)
		out.WriteString(body[:idx])
		out.WriteString(`"env_vars":{"[REDACTED]":"..."}`)
		body = body[end:]
	}
	if out.Len() == 0 {
		return body
	}
	out.WriteString(body)
	return out.String()
}

func isJSONSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

// scanJSONObjectEnd returns the index just past the '}' that closes the
// object opening at body[start] ('{'). Braces inside string values don't
// count, and `\"` / `\\` escapes are honored so a value can't fake a string
// end. If the object never closes (truncated body), the whole tail is treated
// as part of the object: over-redacting beats leaking.
func scanJSONObjectEnd(body string, start int) int {
	depth := 0
	inStr := false
	for i := start; i < len(body); i++ {
		c := body[i]
		if inStr {
			switch c {
			case '\\':
				i++ // skip the escaped byte, whatever it is
			case '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return i + 1
			}
		}
	}
	return len(body)
}

// Truncate returns at most maxLen characters of s, appending "..." if cut.
func Truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// FmtInt is a tiny helper for callers that build flag args like "--max-turns N".
func FmtInt(n int) string { return fmt.Sprintf("%d", n) }
