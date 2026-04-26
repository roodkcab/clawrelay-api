package openai

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
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

// envVarsLogRedactor masks env_vars values in log output, since they may
// contain user-provided secrets.
var envVarsLogRedactor = regexp.MustCompile(`"env_vars"\s*:\s*\{[^}]*\}`)

// SanitizeEnvVarsInLog redacts env_vars before request bodies hit the log file.
func SanitizeEnvVarsInLog(body string) string {
	return envVarsLogRedactor.ReplaceAllString(body, `"env_vars":"[REDACTED]"`)
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
