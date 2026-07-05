package main

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"strings"

	"clawrelay-api/pkg/openai"
)

// V3 usage metering source: the interactive claude (PTY) writes every API
// round's usage into its transcript at
// ~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl. Empirically verified
// facts this file relies on:
//
//   - assistant rows are top-level {"type":"assistant","requestId":...,
//     "isSidechain":...,"sessionId":...} with message.model and
//     message.usage = {input_tokens, cache_creation_input_tokens,
//     cache_read_input_tokens, output_tokens, ...}.
//   - CRITICAL: one assistant message is split into 3-5 jsonl rows (one per
//     content block) with the usage object repeated VERBATIM and the SAME
//     requestId — dedup by requestId or the turn is over-counted 3-5x.
//   - isSidechain=true rows are Task sub-agent calls: real billed API rounds,
//     counted like any other (the requestId dedup rule is universal).
//   - the transcript has no per-model cost figures → CostUSD stays 0.
//
// The cwd encoding of the transcript directory is NOT reimplemented here; the
// caller locates the file with a glob (~/.claude/projects/*/<uuid>.jsonl).

// v3TranscriptLine is the minimal shape of one transcript jsonl row needed for
// metering. Kept local to this file on purpose (types.go stays untouched).
type v3TranscriptLine struct {
	Type      string `json:"type"`
	RequestID string `json:"requestId"`
	Message   *struct {
		Model string `json:"model"`
		Usage *struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
		} `json:"usage"`
	} `json:"message"`
}

// readV3UsageWindow reads the transcript increment [offset, EOF), dedups
// assistant rows by requestId (against prevReqIDs AND within this window) and
// aggregates each surviving row's message.usage per model. It returns the
// per-model aggregate, the new offset, and the set of requestIds seen in THIS
// window (the caller merges them into its cumulative set — a message's
// duplicate rows can straddle a window boundary).
//
// Incremental-tail semantics:
//   - a trailing line without '\n' (a write in flight) is NOT consumed:
//     newOffset stops right before it, so the next window rereads it complete.
//   - a file shorter than offset means rotation/truncation → reread from 0.
//   - rows that are not assistant rows, or have no message.usage, are skipped;
//     an empty message.model is attributed to "unknown".
//
// CostUSD is always 0: the transcript carries no per-model cost.
func readV3UsageWindow(path string, offset int64, prevReqIDs map[string]struct{}) (perModel map[string]openai.TokenCounts, newOffset int64, reqIDs map[string]struct{}, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, offset, nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, offset, nil, err
	}
	if offset < 0 || fi.Size() < offset {
		offset = 0 // rotation/truncation: start over
	}
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, offset, nil, err
	}

	perModel = make(map[string]openai.TokenCounts)
	reqIDs = make(map[string]struct{})
	newOffset = offset

	// Rows can be very large (full content blocks), so no Scanner with its
	// default buffer cap: ReadString on a bufio.Reader grows as needed.
	rd := bufio.NewReaderSize(f, 64*1024)
	for {
		line, rerr := rd.ReadString('\n')
		if rerr == io.EOF {
			// Partial trailing line (no '\n'): a write in flight — leave it for
			// the next window. If line == "" this is just a clean EOF.
			break
		}
		if rerr != nil {
			return perModel, newOffset, reqIDs, rerr
		}
		newOffset += int64(len(line))

		// Cheap pre-filter before the (expensive) unmarshal. Transcript rows are
		// compact JSON, so the key:value pair appears without spaces.
		if !strings.Contains(line, `"type":"assistant"`) {
			continue
		}
		var tl v3TranscriptLine
		if json.Unmarshal([]byte(line), &tl) != nil {
			continue
		}
		if tl.Type != "assistant" || tl.Message == nil || tl.Message.Usage == nil {
			continue
		}
		if rid := tl.RequestID; rid != "" {
			if _, seen := prevReqIDs[rid]; seen {
				continue
			}
			if _, seen := reqIDs[rid]; seen {
				continue
			}
			reqIDs[rid] = struct{}{}
		}
		model := tl.Message.Model
		if model == "" {
			model = "unknown"
		}
		u := tl.Message.Usage
		c := perModel[model]
		c.Input += u.InputTokens
		c.Output += u.OutputTokens
		c.CacheCreation += u.CacheCreationInputTokens
		c.CacheRead += u.CacheReadInputTokens
		perModel[model] = c
	}
	return perModel, newOffset, reqIDs, nil
}
