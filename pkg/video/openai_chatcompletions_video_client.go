package video

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

type OpenAIChatCompletionsVideoClient struct {
	BaseURL    string
	APIKey     string
	Model      string
	Endpoint   string
	HTTPClient *http.Client
}

func NewOpenAIChatCompletionsVideoClient(baseURL, apiKey, model, endpoint string) *OpenAIChatCompletionsVideoClient {
	if endpoint == "" {
		endpoint = "/chat/completions"
	}
	return &OpenAIChatCompletionsVideoClient{
		BaseURL:  strings.TrimRight(baseURL, "/"),
		APIKey:   apiKey,
		Model:    model,
		Endpoint: endpoint,
		HTTPClient: &http.Client{
			Timeout: 300 * time.Second,
		},
	}
}

var mp4Re = regexp.MustCompile(`https?://[^\s"'<>]+\.mp4`)
var imgRe = regexp.MustCompile(`https?://[^\s"'<>]+\.(jpg|jpeg|png|webp)`)

func extractFirst(re *regexp.Regexp, s string) (string, bool) {
	m := re.FindString(s)
	return m, m != ""
}

type ccResp struct {
	Choices []struct {
		Message struct {
			Content any `json:"content"`
		} `json:"message"`
		Delta struct {
			Content any `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

// content 可能是 string，也可能是数组结构；这里尽量转成字符串
func anyToString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case []any:
		// 拼起来当文本
		var b strings.Builder
		for _, item := range t {
			switch it := item.(type) {
			case map[string]any:
				// 可能是 {"type":"text","text":"..."} 之类
				if txt, ok := it["text"].(string); ok && txt != "" {
					b.WriteString(txt)
					b.WriteString("\n")
				}
				// 或 {"type":"output_text","text":"..."} / {"content":"..."}
				if txt, ok := it["content"].(string); ok && txt != "" {
					b.WriteString(txt)
					b.WriteString("\n")
				}
			case string:
				b.WriteString(it)
				b.WriteString("\n")
			}
		}
		return b.String()
	default:
		// 兜底：JSON stringify
		j, _ := json.Marshal(v)
		return string(j)
	}
}

// 解析非流式响应 body，返回 mp4/thumbnail
func parseNonStream(body []byte) (mp4URL, thumbURL string, ok bool) {
	s := string(body)

	// 1) 先粗暴从 body 抓 mp4（兼容返回 html/url）
	if m, ok2 := extractFirst(mp4Re, s); ok2 {
		t, _ := extractFirst(imgRe, s)
		return m, t, true
	}

	// 2) 再按 chat.completions JSON 去拿 content
	var r ccResp
	if err := json.Unmarshal(body, &r); err == nil && len(r.Choices) > 0 {
		content := anyToString(r.Choices[0].Message.Content)
		if content == "" {
			content = anyToString(r.Choices[0].Delta.Content)
		}
		if m, ok2 := extractFirst(mp4Re, content); ok2 {
			t, _ := extractFirst(imgRe, content)
			return m, t, true
		}
	}

	return "", "", false
}

// 解析 SSE：逐行读 data: ... ，拼接/扫描出 mp4
func parseSSE(respBody io.Reader) (mp4URL, thumbURL string, ok bool, rawTail string) {
	scanner := bufio.NewScanner(respBody)
	// SSE 行可能很长，调大 buffer
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	var tail strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "[DONE]" {
				break
			}
			// 尝试直接抓 mp4
			if m, ok2 := extractFirst(mp4Re, data); ok2 {
				t, _ := extractFirst(imgRe, data)
				return m, t, true, data
			}
			// 尝试把 json 解析成 chat chunk
			var r ccResp
			if err := json.Unmarshal([]byte(data), &r); err == nil && len(r.Choices) > 0 {
				content := anyToString(r.Choices[0].Delta.Content)
				if content == "" {
					content = anyToString(r.Choices[0].Message.Content)
				}
				if content != "" {
					tail.WriteString(content)
					tail.WriteString("\n")
					if m, ok2 := extractFirst(mp4Re, content); ok2 {
						t, _ := extractFirst(imgRe, content)
						return m, t, true, content
					}
				}
			} else {
				// 不是标准 chunk，就累计，最后整体抓一次
				tail.WriteString(data)
				tail.WriteString("\n")
			}
		}
	}
	raw := tail.String()
	if m, ok2 := extractFirst(mp4Re, raw); ok2 {
		t, _ := extractFirst(imgRe, raw)
		return m, t, true, raw
	}
	return "", "", false, raw
}

func (c *OpenAIChatCompletionsVideoClient) GenerateVideo(imageURL, prompt string, opts ...VideoOption) (*VideoResult, error) {
	options := &VideoOptions{
		Duration:    6,
		AspectRatio: "16:9",
	}
	for _, opt := range opts {
		opt(options)
	}

	model := c.Model
	if options.Model != "" {
		model = options.Model
	}

	// 组装 chat.completions 请求（兼容 Grok2API：video_config）
	reqBody := map[string]any{
		"model":  model,
		"stream": true, // 你 grok2api 之前是 SSE 进度流，这里默认开；若你确认非流式也行，可改 false
		"messages": []map[string]any{
			{"role": "user", "content": prompt},
		},
		"video_config": map[string]any{
			"aspect_ratio": options.AspectRatio,
			"video_length": options.Duration,
		},
	}

	// 图生视频：用多模态 content
	if imageURL != "" {
		reqBody["messages"] = []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": prompt},
					{"type": "image_url", "image_url": map[string]any{"url": imageURL}},
				},
			},
		}
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := c.BaseURL + c.Endpoint
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	ct := resp.Header.Get("Content-Type")

	// SSE：text/event-stream
	if strings.Contains(ct, "text/event-stream") {
		mp4, thumb, ok, raw := parseSSE(resp.Body)
		if !ok {
			return nil, fmt.Errorf("no mp4 url found in SSE stream, tail=%s", raw)
		}
		return &VideoResult{
			Status:       "completed",
			Completed:    true,
			VideoURL:     mp4,
			ThumbnailURL: thumb,
			Duration:     options.Duration,
		}, nil
	}

	// 非流式 JSON
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	mp4, thumb, ok := parseNonStream(body)
	if !ok {
		return nil, fmt.Errorf("no mp4 url found in response: %s", string(body))
	}

	return &VideoResult{
		Status:       "completed",
		Completed:    true,
		VideoURL:     mp4,
		ThumbnailURL: thumb,
		Duration:     options.Duration,
	}, nil
}

func (c *OpenAIChatCompletionsVideoClient) GetTaskStatus(taskID string) (*VideoResult, error) {
	// 同步模式不需要轮询
	return nil, fmt.Errorf("GetTaskStatus not supported for chat.completions video client")
}
