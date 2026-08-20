package seahorse

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/sipeed/picoclaw/pkg/tools"
)

// SemanticTool searches stored conversation history by meaning using
// embedding-based retrieval (complements short_grep's keyword search).
type SemanticTool struct {
	engine *RetrievalEngine
}

func NewSemanticTool(engine *RetrievalEngine) *SemanticTool {
	return &SemanticTool{engine: engine}
}

func (t *SemanticTool) Name() string {
	return "short_semantic"
}

func (t *SemanticTool) Description() string {
	return `Search stored conversation history by MEANING (semantic/embedding retrieval).

Use this when short_grep (keyword search) can't find something because the
wording differs — e.g. "how do I fix the audio crackle" will find past
conversations about "rtkit missing causes constant crackling" even without
shared keywords.

Searches across ALL conversations by default (it is a memory-recall tool).
Set all_conversations: false to restrict to the current chat.

Parameters:
- query (required): What you are trying to recall, in your own words
- limit: Max results (default: 8)
- min_score: Cosine similarity threshold, 0..1 (default: 0.35). Raise for
  stricter matches, lower for fuzzier recall.
- all_conversations: Search all conversations (default: true)

Returns:
{
  "success": true,
  "messages": [{"id": "123", "snippet": "...", "role": "user", "conversationId": 1, "rank": 0.72}],
  "hint": "No semantic matches above the score threshold. Try lowering min_score..."
}

Rank field: cosine similarity (0..1, higher = more similar). Expand with
short_expand using the message "id" to read full context.

Examples:
  {"query": "gpu wedges and resets under compute load"}
  {"query": "how do we back up the workspace", "min_score": 0.5}
  {"query": "pi hole admin password", "all_conversations": true, "limit": 5}`
}

func (t *SemanticTool) Parameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "What you are trying to recall, in your own words",
			},
			"limit": map[string]any{
				"type":        "integer",
				"description": "Maximum number of results (default: 8)",
			},
			"min_score": map[string]any{
				"type":        "number",
				"description": "Cosine similarity threshold 0..1 (default: 0.35). Higher = stricter.",
			},
			"all_conversations": map[string]any{
				"type":        "boolean",
				"description": "Search all conversations (default: true). Set false to scope to current chat.",
			},
		},
		"required": []string{"query"},
	}
}

func (t *SemanticTool) Execute(ctx context.Context, args map[string]any) *tools.ToolResult {
	query, ok := args["query"].(string)
	if !ok || strings.TrimSpace(query) == "" {
		return tools.ErrorResult("Missing required 'query' argument. Example: {\"query\": \"how did we fix the audio crackle\"}")
	}

	input := SemanticInput{
		Query:            query,
		AllConversations: true, // semantic memory is a recall tool by default
	}

	if limit, ok := args["limit"].(float64); ok && limit > 0 {
		input.Limit = int(limit)
	}
	if minScore, ok := args["min_score"].(float64); ok {
		input.MinScore = minScore
	}
	if allConv, ok := args["all_conversations"].(bool); ok {
		input.AllConversations = allConv
	}

	// Scope to current conversation when requested (default is all).
	if !input.AllConversations {
		if sessionKey := tools.ToolSessionKey(ctx); sessionKey != "" {
			if conv, err := t.engine.store.GetConversationBySessionKey(ctx, sessionKey); err == nil && conv != nil {
				input.ConversationID = conv.ConversationID
			}
		}
	}

	result, err := t.engine.Semantic(ctx, input)
	if err != nil {
		return tools.ErrorResult("Semantic search failed: " + err.Error())
	}

	output := map[string]any{
		"success":  result.Success,
		"messages": result.Messages,
	}
	if result.Hint != "" {
		output["hint"] = result.Hint
	}

	data, err := json.Marshal(output)
	if err != nil {
		return tools.ErrorResult(fmt.Sprintf("failed to marshal semantic result: %v", err))
	}
	return tools.NewToolResult(string(data))
}
