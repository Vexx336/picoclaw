package config

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Ported from the pre-M3 dirty binary: named agent-tree profiles +
// active_profile selection (launcher agent-tree selector).
func TestAgentsConfigProfilesDecode(t *testing.T) {
	var cfg AgentsConfig
	err := json.Unmarshal([]byte(`{
		"defaults": {"model_name": "base"},
		"list": [{"id": "main"}],
		"profiles": {
			"flash": {
				"defaults": {"model_name": "openrouter-flash-byok"},
				"list": [{"id": "main"}, {"id": "pro"}],
				"description": "cheap"
			},
			"pro": {
				"defaults": {"model_name": "openrouter-pro-byok"}
			}
		},
		"active_profile": "flash"
	}`), &cfg)
	require.NoError(t, err)
	assert.Equal(t, "flash", cfg.ActiveProfile)
	require.Len(t, cfg.Profiles, 2)
	assert.Equal(t, "cheap", cfg.Profiles["flash"].Description)
	assert.Equal(t, "openrouter-pro-byok", cfg.Profiles["pro"].Defaults.ModelName)
}

func TestAgentsConfigApplyProfile(t *testing.T) {
	cfg := AgentsConfig{
		Defaults: AgentDefaults{ModelName: "base"},
		List:     []AgentConfig{{ID: "main"}},
		Profiles: map[string]AgentsConfig{
			"flash": {
				Defaults: AgentDefaults{ModelName: "openrouter-flash-byok"},
				List:     []AgentConfig{{ID: "main"}, {ID: "pro"}},
			},
		},
		ActiveProfile: "",
	}

	assert.True(t, cfg.ApplyProfile("flash"))
	assert.Equal(t, "openrouter-flash-byok", cfg.Defaults.ModelName)
	assert.Len(t, cfg.List, 2)
	assert.Equal(t, "flash", cfg.ActiveProfile)

	// Missing profile: no-op.
	assert.False(t, cfg.ApplyProfile("nope"))
	assert.Equal(t, "openrouter-flash-byok", cfg.Defaults.ModelName)
	assert.Equal(t, "flash", cfg.ActiveProfile)
}

func TestAgentsConfigCloneIsDeep(t *testing.T) {
	cfg := AgentsConfig{
		Defaults: AgentDefaults{ModelName: "base"},
		List:     []AgentConfig{{ID: "main"}},
		Dispatch: &DispatchConfig{Rules: []DispatchRule{{Name: "r1", Agent: "main"}}},
		Profiles: map[string]AgentsConfig{
			"flash": {List: []AgentConfig{{ID: "pro"}}},
		},
	}
	cp := cfg.Clone()
	cp.List[0].ID = "mutated"
	cp.Dispatch.Rules[0].Name = "mutated"
	cp.Profiles["flash"].List[0].ID = "mutated"

	assert.Equal(t, "main", cfg.List[0].ID)
	assert.Equal(t, "r1", cfg.Dispatch.Rules[0].Name)
	assert.Equal(t, "pro", cfg.Profiles["flash"].List[0].ID)
}
