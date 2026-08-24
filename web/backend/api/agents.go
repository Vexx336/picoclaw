package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/logger"
)

// registerAgentTreeRoutes binds agent-tree / profile management endpoints.
// These back the launcher Advanced Settings "Agent Trees" editor:
//
//	GET    /api/agents/tree      → live tree + named profiles + active_profile
//	POST   /api/agents/apply     → copy a named profile into the live agents block
//	PUT    /api/agents/profile   → create/update a named profile
//	DELETE /api/agents/profile   → remove a named profile
func (h *Handler) registerAgentTreeRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/agents/tree", h.handleGetAgentTree)
	mux.HandleFunc("POST /api/agents/apply", h.handleApplyAgentProfile)
	mux.HandleFunc("PUT /api/agents/profile", h.handleUpsertAgentProfile)
	mux.HandleFunc("DELETE /api/agents/profile", h.handleDeleteAgentProfile)
}

// agentTreeResponse is the compact agent-tree view returned by the launcher.
type agentTreeResponse struct {
	Live          config.AgentsConfig          `json:"live"`
	Profiles      map[string]config.AgentsConfig `json:"profiles"`
	ActiveProfile string                        `json:"active_profile"`
}

// handleGetAgentTree returns the live agents block plus all named profiles.
func (h *Handler) handleGetAgentTree(w http.ResponseWriter, r *http.Request) {
	cfg, err := config.LoadConfig(h.configPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to load config: %v", err), http.StatusInternalServerError)
		return
	}
	resp := agentTreeResponse{
		Live:          cfg.Agents,
		Profiles:      cfg.Agents.Profiles,
		ActiveProfile: cfg.Agents.ActiveProfile,
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

// handleApplyAgentProfile copies the named profile into the live agents block
// (defaults + list + dispatch) and records it as active_profile.
//
//	POST /api/agents/apply {"profile":"flash"}
func (h *Handler) handleApplyAgentProfile(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Profile string `json:"profile"`
	}
	if err := decodeJSONBody(w, r, &req); err != nil {
		return
	}
	if req.Profile == "" {
		http.Error(w, "profile is required", http.StatusBadRequest)
		return
	}

	cfg, err := config.LoadConfig(h.configPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to load config: %v", err), http.StatusInternalServerError)
		return
	}
	if !cfg.Agents.ApplyProfile(req.Profile) {
		http.Error(w, fmt.Sprintf("profile %q not found", req.Profile), http.StatusNotFound)
		return
	}
	if !h.saveAgentConfig(w, cfg) {
		return
	}
	writeAgentTreeOK(w, map[string]any{
		"status":         "ok",
		"active_profile": req.Profile,
	})
}

// handleUpsertAgentProfile creates or updates a named profile. The payload's
// tree is a full AgentsConfig (defaults + list + dispatch + description).
// Nested profiles/active_profile in the payload are ignored. When the profile
// is currently active, the live agents block is re-applied from it so the
// running tree stays in sync with the saved profile.
//
//	PUT /api/agents/profile {"name":"flash","tree":{...}}
func (h *Handler) handleUpsertAgentProfile(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name string                `json:"name"`
		Tree *config.AgentsConfig  `json:"tree"`
	}
	if err := decodeJSONBody(w, r, &req); err != nil {
		return
	}
	if req.Name == "" {
		http.Error(w, "name is required", http.StatusBadRequest)
		return
	}
	if req.Tree == nil {
		http.Error(w, "tree is required", http.StatusBadRequest)
		return
	}

	cfg, err := config.LoadConfig(h.configPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to load config: %v", err), http.StatusInternalServerError)
		return
	}
	if cfg.Agents.Profiles == nil {
		cfg.Agents.Profiles = make(map[string]config.AgentsConfig)
	}
	tree := *req.Tree
	// Sanitize: profiles can't nest profiles or carry active_profile.
	tree.Profiles = nil
	tree.ActiveProfile = ""
	cfg.Agents.Profiles[req.Name] = tree
	if cfg.Agents.ActiveProfile == req.Name {
		cfg.Agents.ApplyProfile(req.Name)
	}
	if !h.saveAgentConfig(w, cfg) {
		return
	}
	writeAgentTreeOK(w, map[string]any{
		"status":         "ok",
		"active_profile": cfg.Agents.ActiveProfile,
	})
}

// handleDeleteAgentProfile removes a named profile. If the removed profile was
// active, active_profile is cleared (the live agents block is left untouched).
//
//	DELETE /api/agents/profile {"name":"flash"}
func (h *Handler) handleDeleteAgentProfile(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name string `json:"name"`
	}
	if err := decodeJSONBody(w, r, &req); err != nil {
		return
	}
	if req.Name == "" {
		http.Error(w, "name is required", http.StatusBadRequest)
		return
	}

	cfg, err := config.LoadConfig(h.configPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to load config: %v", err), http.StatusInternalServerError)
		return
	}
	if _, ok := cfg.Agents.Profiles[req.Name]; !ok {
		http.Error(w, fmt.Sprintf("profile %q not found", req.Name), http.StatusNotFound)
		return
	}
	delete(cfg.Agents.Profiles, req.Name)
	if cfg.Agents.ActiveProfile == req.Name {
		cfg.Agents.ActiveProfile = ""
	}
	if !h.saveAgentConfig(w, cfg) {
		return
	}
	writeAgentTreeOK(w, map[string]any{
		"status":         "ok",
		"active_profile": cfg.Agents.ActiveProfile,
	})
}

// saveAgentConfig runs the same validation/save pipeline as handlePatchConfig
// for an in-memory Config that was mutated by an agent-tree endpoint. It writes
// an HTTP error and returns false when validation or persistence fails.
func (h *Handler) saveAgentConfig(w http.ResponseWriter, cfg *config.Config) bool {
	if err := cfg.SecurityCopyFrom(h.configPath); err != nil {
		http.Error(w, fmt.Sprintf("Failed to apply security config: %v", err), http.StatusInternalServerError)
		return false
	}
	if errs := validateConfig(cfg); len(errs) > 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]any{
			"status": "validation_error",
			"errors": errs,
		})
		return false
	}
	if err := config.SaveConfig(h.configPath, cfg); err != nil {
		http.Error(w, fmt.Sprintf("Failed to save config: %v", err), http.StatusInternalServerError)
		return false
	}
	h.applyRuntimeLogLevel()
	logger.Infof("configuration updated successfully")
	return true
}

// decodeJSONBody reads and unmarshals a request body (max 1MB). It writes an
// HTTP error and returns an error on failure.
func decodeJSONBody(w http.ResponseWriter, r *http.Request, out any) error {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return err
	}
	defer r.Body.Close()
	if err := json.Unmarshal(body, out); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return err
	}
	return nil
}

// writeAgentTreeOK sends a JSON success response.
func writeAgentTreeOK(w http.ResponseWriter, payload map[string]any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(payload)
}
