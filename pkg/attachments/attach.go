// Package attachments handles base64 image/file payloads inside chat messages.
// The relay decodes these once and writes them to disk so the upstream CLI
// (which can read files but not parse multipart JSON) sees stable paths.
package attachments

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// Saved describes one decoded attachment on disk.
type Saved struct {
	Path    string
	IsImage bool
}

// MimeToExt picks a sensible extension for common content-types.
func MimeToExt(mime string) string {
	switch {
	case strings.Contains(mime, "jpeg") || strings.Contains(mime, "jpg"):
		return ".jpg"
	case strings.Contains(mime, "png"):
		return ".png"
	case strings.Contains(mime, "gif"):
		return ".gif"
	case strings.Contains(mime, "webp"):
		return ".webp"
	case strings.Contains(mime, "pdf"):
		return ".pdf"
	case strings.Contains(mime, "csv"):
		return ".csv"
	case strings.Contains(mime, "json"):
		return ".json"
	case strings.Contains(mime, "html"):
		return ".html"
	case strings.Contains(mime, "xml"):
		return ".xml"
	case strings.Contains(mime, "zip"):
		return ".zip"
	case strings.Contains(mime, "plain") || strings.Contains(mime, "text"):
		return ".txt"
	case strings.Contains(mime, "markdown"):
		return ".md"
	default:
		return ".bin"
	}
}

// ExtractAndSave walks a message's `content` field looking for image_url and
// file_url parts encoded as `data:` URIs. Each decoded payload is written to
// disk and returned. When sessionDir is non-empty, files are content-hashed
// for cross-turn dedup; otherwise each call gets a fresh /tmp filename and
// the caller should delete the files after use.
func ExtractAndSave(content json.RawMessage, sessionDir, prefixImage, prefixFile string) []Saved {
	if len(content) == 0 {
		return nil
	}
	var parts []struct {
		Type     string `json:"type"`
		ImageURL *struct {
			URL string `json:"url"`
		} `json:"image_url,omitempty"`
		FileURL *struct {
			URL      string `json:"url"`
			Filename string `json:"filename,omitempty"`
		} `json:"file_url,omitempty"`
	}
	if err := json.Unmarshal(content, &parts); err != nil {
		return nil
	}

	if sessionDir != "" {
		os.MkdirAll(sessionDir, 0755)
	}

	if prefixImage == "" {
		prefixImage = "img"
	}
	if prefixFile == "" {
		prefixFile = "file"
	}

	var files []Saved
	for _, p := range parts {
		var dataURL string
		var isImage bool
		var prefix string

		switch {
		case p.Type == "image_url" && p.ImageURL != nil:
			dataURL = p.ImageURL.URL
			isImage = true
			prefix = prefixImage
		case p.Type == "file_url" && p.FileURL != nil:
			dataURL = p.FileURL.URL
			isImage = false
			prefix = prefixFile
		default:
			continue
		}

		if !strings.HasPrefix(dataURL, "data:") {
			continue
		}
		comma := strings.Index(dataURL, ",")
		if comma < 0 {
			continue
		}
		header := dataURL[5:comma]
		b64Data := dataURL[comma+1:]

		ext := MimeToExt(header)
		if p.Type == "file_url" && p.FileURL != nil && p.FileURL.Filename != "" {
			if dot := strings.LastIndex(p.FileURL.Filename, "."); dot >= 0 {
				ext = p.FileURL.Filename[dot:]
			}
		}

		dir := "/tmp"
		if sessionDir != "" {
			dir = sessionDir
		}

		var filePath string
		if sessionDir != "" {
			// Hash the base64 body so duplicate uploads across turns reuse
			// the same file (no redundant decode/write).
			hash := sha256.Sum256([]byte(b64Data))
			name := fmt.Sprintf("%s-%s%s", prefix, hex.EncodeToString(hash[:8]), ext)
			filePath = filepath.Join(dir, name)
			if _, err := os.Stat(filePath); err == nil {
				files = append(files, Saved{Path: filePath, IsImage: isImage})
				continue
			}
		} else {
			var randBytes [8]byte
			if _, err := rand.Read(randBytes[:]); err != nil {
				continue
			}
			filePath = filepath.Join(dir, fmt.Sprintf("%s-%s%s", prefix, hex.EncodeToString(randBytes[:]), ext))
		}

		fileBytes, err := base64.StdEncoding.DecodeString(b64Data)
		if err != nil {
			log.Printf("Failed to decode attachment base64: %v", err)
			continue
		}
		if err := os.WriteFile(filePath, fileBytes, 0600); err != nil {
			log.Printf("Failed to write file %s: %v", filePath, err)
			continue
		}
		log.Printf("Saved attachment to: %s", filePath)
		files = append(files, Saved{Path: filePath, IsImage: isImage})
	}
	return files
}
