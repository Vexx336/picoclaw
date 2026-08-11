package agent

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers"
)

var resolvedImagePathTagRegex = regexp.MustCompile(`\[image:[^\s\]][^\]]*\]`)

func messagesContainMedia(messages []providers.Message) bool {
	for _, msg := range messages {
		for _, ref := range msg.Media {
			if strings.TrimSpace(ref) != "" {
				return true
			}
		}
	}
	return false
}

func stripMessageMedia(messages []providers.Message) []providers.Message {
	if !messagesContainMedia(messages) {
		return messages
	}
	stripped := make([]providers.Message, len(messages))
	for i, msg := range messages {
		stripped[i] = msg
		stripped[i].Media = nil
	}
	return stripped
}

func isVisionUnsupportedError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())

	// OpenRouter (and OpenAI-compatible) style.
	if strings.Contains(msg, "no endpoints found that support image input") {
		return true
	}

	// Common provider variants.
	if strings.Contains(msg, "does not support image input") ||
		strings.Contains(msg, "does not support image inputs") ||
		strings.Contains(msg, "does not support images") ||
		strings.Contains(msg, "image input is not supported") ||
		strings.Contains(msg, "images are not supported") ||
		strings.Contains(msg, "does not support vision") ||
		strings.Contains(msg, "unsupported content type: image_url") {
		return true
	}

	// Some providers return a generic "invalid" message that still mentions image_url.
	if strings.Contains(msg, "image_url") && strings.Contains(msg, "invalid") {
		return true
	}

	// DeepSeek and other strict providers reject the image_url field at the
	// JSON schema level with an "unknown variant" error rather than a semantic
	// "not supported" message.
	if strings.Contains(msg, "unknown variant") && strings.Contains(msg, "image_url") {
		return true
	}

	return false
}

func visionUnsupportedModelError(modelName string, imageModelConfigured bool) error {
	modelName = strings.TrimSpace(modelName)
	guidance := "update agents.defaults.image_model (or image_model_fallbacks) to a multimodal model such as gpt-5.4, gpt-4o, claude-sonnet-4.6, or gemini-2.0-flash"
	if imageModelConfigured {
		if modelName != "" {
			return fmt.Errorf(
				"selected vision model %q does not support image input; %s",
				modelName,
				guidance,
			)
		}
		return fmt.Errorf(
			"selected vision model does not support image input; %s",
			guidance,
		)
	}
	if modelName != "" {
		return fmt.Errorf(
			"active model %q does not support image input; %s",
			modelName,
			guidance,
		)
	}
	return fmt.Errorf(
		"the active model does not support image input; %s",
		guidance,
	)
}

func sameCandidateSet(a, b []providers.FallbackCandidate) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].StableKey() != b[i].StableKey() {
			return false
		}
	}
	return true
}

func messagesContainCurrentTurnMediaTurn(messages []providers.Message) bool {
	for _, msg := range messages {
		if len(msg.Media) > 0 {
			return true
		}
		if resolvedImagePathTagRegex.MatchString(msg.Content) {
			return true
		}
	}
	return false
}

func (p *Pipeline) routeMediaTurn(ts *turnState, exec *turnExecution) error {
	if p == nil || ts == nil || ts.agent == nil || exec == nil ||
		!messagesContainCurrentTurnMediaTurn(currentTurnMessages(exec.callMessages, exec.currentTurnStart)) {
		return nil
	}

	var targetCandidates []providers.FallbackCandidate
	var targetModelName string
	var routeReason string

	switch {
	case len(ts.agent.ImageCandidates) > 0:
		targetCandidates = append([]providers.FallbackCandidate(nil), ts.agent.ImageCandidates...)
		targetModelName = strings.TrimSpace(p.Cfg.Agents.Defaults.ImageModel)
		routeReason = "configured_image_model"
	case exec.usedLight && len(ts.agent.Candidates) > 0:
		targetCandidates = append([]providers.FallbackCandidate(nil), ts.agent.Candidates...)
		targetModelName = strings.TrimSpace(ts.agent.Model)
		routeReason = "bypass_light_model_for_media"
	default:
		return nil
	}

	if len(targetCandidates) == 0 {
		return nil
	}

	targetModel := resolvedCandidateModel(targetCandidates, targetModelName)
	targetProvider := exec.activeProvider
	firstCandidate := targetCandidates[0]
	if provider, err := providerForFallbackCandidate(
		ts.agent,
		ts.agent.Provider,
		ts.agent.Candidates,
		firstCandidate,
	); err != nil {
		return err
	} else if provider != nil {
		targetProvider = provider
	}

	resolvedModelName := resolvedCandidateModelName(targetCandidates, targetModelName)
	if sameCandidateSet(exec.activeCandidates, targetCandidates) &&
		exec.activeModel == targetModel &&
		exec.llmModelName == resolvedModelName {
		return nil
	}

	exec.activeCandidates = targetCandidates
	exec.activeModel = targetModel
	exec.activeProvider = targetProvider
	exec.activeModelConfig = resolveActiveModelConfig(
		p.Cfg,
		ts.agent.Workspace,
		targetCandidates,
		targetModel,
		p.Cfg.Agents.Defaults.Provider,
	)
	exec.llmModelName = resolvedModelName
	exec.usedLight = false

	logger.InfoCF("agent", "Media turn routing selected model", map[string]any{
		"agent_id":       ts.agent.ID,
		"reason":         routeReason,
		"model":          exec.activeModel,
		"model_name":     exec.llmModelName,
		"candidates":     len(exec.activeCandidates),
		"messages_count": len(exec.callMessages),
	})

	return nil
}

func modelConfigLooksVisionCapable(mc *config.ModelConfig) bool {
	if mc == nil {
		return false
	}
	parts := []string{
		strings.TrimSpace(mc.ModelName),
		strings.TrimSpace(mc.Model),
		strings.TrimSpace(mc.Provider),
	}
	combined := strings.ToLower(strings.Join(parts, " "))
	if combined == "" {
		return false
	}
	visionHints := []string{
		"gpt-4o",
		"gpt-4.1",
		"gpt-5",
		"claude-3",
		"claude-opus-4",
		"claude-sonnet-4",
		"gemini",
		"vision",
		"multimodal",
		"vl",
	}
	for _, hint := range visionHints {
		if strings.Contains(combined, hint) {
			return true
		}
	}
	return false
}

func (p *Pipeline) recoverVisionRouting(ts *turnState, exec *turnExecution) bool {
	if p == nil || p.Cfg == nil || ts == nil || ts.agent == nil || exec == nil {
		return false
	}
	if !messagesContainCurrentTurnMediaTurn(currentTurnMessages(exec.callMessages, exec.currentTurnStart)) {
		return false
	}

	currentKey := ""
	if len(exec.activeCandidates) > 0 {
		currentKey = exec.activeCandidates[0].StableKey()
	} else {
		currentKey = providers.ModelKey(
			resolvedCandidateProvider(exec.activeCandidates, p.Cfg.Agents.Defaults.Provider),
			resolvedCandidateModel(exec.activeCandidates, exec.activeModel),
		)
	}

	candidateNames := make([]string, 0, 8)
	seenNames := make(map[string]bool)
	addName := func(name string) {
		name = strings.TrimSpace(name)
		if name == "" || seenNames[name] {
			return
		}
		seenNames[name] = true
		candidateNames = append(candidateNames, name)
	}

	for _, name := range p.Cfg.Agents.Defaults.ImageModelFallbacks {
		addName(name)
	}
	for _, mc := range p.Cfg.ModelList {
		if !modelConfigLooksVisionCapable(mc) {
			continue
		}
		addName(strings.TrimSpace(mc.ModelName))
	}

	for _, name := range candidateNames {
		resolved := resolveModelCandidates(
			p.Cfg,
			p.Cfg.Agents.Defaults.Provider,
			name,
			nil,
		)
		if len(resolved) == 0 {
			continue
		}
		if resolved[0].StableKey() == currentKey {
			continue
		}
		provider, ok := ts.agent.CandidateProviders[resolved[0].StableKey()]
		if !ok || provider == nil {
			populateCandidateProvidersFromNames(p.Cfg, ts.agent.Workspace, []string{name}, ts.agent.CandidateProviders)
			provider = ts.agent.CandidateProviders[resolved[0].StableKey()]
		}
		if provider == nil {
			continue
		}

		exec.activeCandidates = resolved
		exec.activeModel = resolvedCandidateModel(resolved, name)
		exec.activeProvider = provider
		exec.activeModelConfig = resolveActiveModelConfig(
			p.Cfg,
			ts.agent.Workspace,
			resolved,
			exec.activeModel,
			p.Cfg.Agents.Defaults.Provider,
		)
		exec.llmModelName = resolvedCandidateModelName(resolved, name)
		exec.llmModel = exec.activeModel
		exec.usedLight = false

		logger.WarnCF("agent", "Recovered media turn by switching to alternate multimodal model", map[string]any{
			"agent_id":       ts.agent.ID,
			"recovered_from": currentKey,
			"recovered_to":   resolved[0].StableKey(),
			"model_name":     exec.llmModelName,
		})
		return true
	}

	return false
}
