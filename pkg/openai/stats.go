package openai

import (
	"sync"
	"time"
)

// ModelTokenStats is one model's lifetime accumulator.
type ModelTokenStats struct {
	Requests      int64   `json:"requests"`
	Input         int64   `json:"input_tokens"`
	Output        int64   `json:"output_tokens"`
	CacheCreation int64   `json:"cache_creation_input_tokens"`
	CacheRead     int64   `json:"cache_read_input_tokens"`
	Total         int64   `json:"total_tokens"`
	CostUSD       float64 `json:"cost_usd"`
}

// TokenStatsSnapshot is the JSON shape served by /v1/stats.
type TokenStatsSnapshot struct {
	TotalRequests int64                       `json:"total_requests"`
	InputTokens   int64                       `json:"input_tokens"`
	OutputTokens  int64                       `json:"output_tokens"`
	CacheCreation int64                       `json:"cache_creation_input_tokens"`
	CacheRead     int64                       `json:"cache_read_input_tokens"`
	TotalTokens   int64                       `json:"total_tokens"`
	CostUSD       float64                     `json:"cost_usd"`
	PerModel      map[string]*ModelTokenStats `json:"per_model"`
	StartTime     string                      `json:"start_time"`
	Uptime        string                      `json:"uptime"`
}

// Stats is a thread-safe lifetime token counter shared by all handlers
// inside a single relay binary.
type Stats struct {
	mu            sync.Mutex
	requests      int64
	input         int64
	output        int64
	cacheCreation int64
	cacheRead     int64
	costUSD       float64
	perModel      map[string]*ModelTokenStats
	startTime     time.Time
}

func NewStats() *Stats {
	return &Stats{
		perModel:  make(map[string]*ModelTokenStats),
		startTime: time.Now(),
	}
}

// Record registers one completed turn's token usage and cost against a model.
func (s *Stats) Record(model string, input, output, cacheCreation, cacheRead int, costUSD float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests++
	s.input += int64(input)
	s.output += int64(output)
	s.cacheCreation += int64(cacheCreation)
	s.cacheRead += int64(cacheRead)
	s.costUSD += costUSD

	ms, ok := s.perModel[model]
	if !ok {
		ms = &ModelTokenStats{}
		s.perModel[model] = ms
	}
	ms.Requests++
	ms.Input += int64(input)
	ms.Output += int64(output)
	ms.CacheCreation += int64(cacheCreation)
	ms.CacheRead += int64(cacheRead)
	ms.Total += int64(input + output + cacheCreation + cacheRead)
	ms.CostUSD += costUSD
}

// Snapshot returns a deep-copied view safe for marshaling.
func (s *Stats) Snapshot() TokenStatsSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	pm := make(map[string]*ModelTokenStats, len(s.perModel))
	for k, v := range s.perModel {
		cp := *v
		pm[k] = &cp
	}
	return TokenStatsSnapshot{
		TotalRequests: s.requests,
		InputTokens:   s.input,
		OutputTokens:  s.output,
		CacheCreation: s.cacheCreation,
		CacheRead:     s.cacheRead,
		TotalTokens:   s.input + s.output + s.cacheCreation + s.cacheRead,
		CostUSD:       s.costUSD,
		PerModel:      pm,
		StartTime:     s.startTime.Format(time.RFC3339),
		Uptime:        time.Since(s.startTime).Truncate(time.Second).String(),
	}
}

// BuildUsageInfo produces an OpenAI-shaped UsageInfo from raw counters.
// Convention: input is "uncached" prompt tokens, cacheRead is the cached
// portion. The OpenAI spec puts the *total* input in PromptTokens.
func BuildUsageInfo(input, output, cacheRead, cacheCreation int) *UsageInfo {
	promptTokens := input + cacheRead + cacheCreation
	if promptTokens == 0 && output == 0 {
		return nil
	}
	u := &UsageInfo{
		PromptTokens:     promptTokens,
		CompletionTokens: output,
		TotalTokens:      promptTokens + output,
	}
	if cacheRead > 0 || cacheCreation > 0 {
		u.PromptTokensDetails = &PromptTokensDetails{
			CachedTokens:        cacheRead,
			CacheCreationTokens: cacheCreation,
		}
	}
	return u
}
