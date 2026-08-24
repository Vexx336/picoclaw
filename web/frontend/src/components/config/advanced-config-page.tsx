import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"

import { launcherFetch } from "@/api/http"
import { ConfigChangeNotice } from "@/components/config-change-notice"
import { Field, SwitchCardField } from "@/components/shared-form"
import { PageHeader } from "@/components/page-header"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { showSaveSuccessOrRestartToast } from "@/lib/restart-required"
import { refreshGatewayState } from "@/store/gateway"

interface SemanticMemoryConfig {
  enableSemantic: boolean
  embeddingEndpoint: string
  embeddingModel: string
  embeddingDim: number
  topK: number
  minScore: number
  asyncWrite: boolean
  ignoreSessionsForEmbedding: string[]
}

const DEFAULT_SEMANTIC: SemanticMemoryConfig = {
  enableSemantic: false,
  embeddingEndpoint: "",
  embeddingModel: "",
  embeddingDim: 1024,
  topK: 8,
  minScore: 0.35,
  asyncWrite: true,
  ignoreSessionsForEmbedding: [],
}

function readSemantic(raw: unknown): SemanticMemoryConfig {
  if (!raw || typeof raw !== "object") return { ...DEFAULT_SEMANTIC }
  const r = raw as Record<string, unknown>
  return {
    enableSemantic: Boolean(r.enableSemantic ?? false),
    embeddingEndpoint:
      typeof r.embeddingEndpoint === "string" ? r.embeddingEndpoint : "",
    embeddingModel:
      typeof r.embeddingModel === "string" ? r.embeddingModel : "",
    embeddingDim:
      typeof r.embeddingDim === "number" ? r.embeddingDim : DEFAULT_SEMANTIC.embeddingDim,
    topK: typeof r.topK === "number" ? r.topK : DEFAULT_SEMANTIC.topK,
    minScore:
      typeof r.minScore === "number" ? r.minScore : DEFAULT_SEMANTIC.minScore,
    asyncWrite: Boolean(r.asyncWrite ?? true),
    ignoreSessionsForEmbedding: Array.isArray(r.ignoreSessionsForEmbedding)
      ? (r.ignoreSessionsForEmbedding as string[])
      : [],
  }
}

function asInt(v: string, fallback: number): number {
  const n = Number.parseInt(v, 10)
  return Number.isFinite(n) ? n : fallback
}

function asFloat(v: string, fallback: number): number {
  const n = Number.parseFloat(v)
  return Number.isFinite(n) ? n : fallback
}

interface AgentRowDraft {
  id: string
  name: string
  model: string
  skills: string
  isDefault: boolean
}

interface AgentTreeDraft {
  description: string
  modelName: string
  provider: string
  imageModel: string
  maxTokens: string
  contextWindow: string
  temperature: string
  maxToolIterations: string
  agents: AgentRowDraft[]
  dispatchJson: string
}

const EMPTY_TREE: AgentTreeDraft = {
  description: "",
  modelName: "",
  provider: "",
  imageModel: "",
  maxTokens: "",
  contextWindow: "",
  temperature: "",
  maxToolIterations: "",
  agents: [],
  dispatchJson: "[]",
}

function readDispatchJson(raw: unknown): string {
  if (!raw || typeof raw !== "object") return "[]"
  const rules = (raw as Record<string, unknown>).rules
  if (!Array.isArray(rules)) return "[]"
  return JSON.stringify(rules, null, 2)
}

function treeFromRaw(raw: unknown): AgentTreeDraft {
  if (!raw || typeof raw !== "object") return { ...EMPTY_TREE }
  const r = raw as Record<string, any>
  const defaults = r.defaults ?? {}
  const list: any[] = Array.isArray(r.list) ? r.list : []
  return {
    description: typeof r.description === "string" ? r.description : "",
    modelName:
      typeof defaults.model_name === "string" ? defaults.model_name : "",
    provider: typeof defaults.provider === "string" ? defaults.provider : "",
    imageModel:
      typeof defaults.image_model === "string" ? defaults.image_model : "",
    maxTokens:
      defaults.max_tokens != null ? String(defaults.max_tokens) : "",
    contextWindow:
      defaults.context_window != null ? String(defaults.context_window) : "",
    temperature:
      defaults.temperature != null ? String(defaults.temperature) : "",
    maxToolIterations:
      defaults.max_tool_iterations != null
        ? String(defaults.max_tool_iterations)
        : "",
    agents: list.map((a) => ({
      id: typeof a.id === "string" ? a.id : "",
      name: typeof a.name === "string" ? a.name : "",
      model:
        typeof a.model === "string"
          ? a.model
          : typeof a.model?.primary === "string"
            ? a.model.primary
            : "",
      skills: Array.isArray(a.skills) ? a.skills.join(", ") : "",
      isDefault: Boolean(a.default),
    })),
    dispatchJson: readDispatchJson(r.dispatch),
  }
}

function parseDispatchRules(jsonText: string): unknown {
  const t = jsonText.trim()
  if (t === "") return null
  const parsed: unknown = JSON.parse(t)
  if (!Array.isArray(parsed)) {
    throw new Error("dispatch rules must be a JSON array")
  }
  return { rules: parsed }
}

function treeToPatch(draft: AgentTreeDraft): Record<string, unknown> {
  return {
    description: draft.description,
    defaults: {
      model_name: draft.modelName.trim() === "" ? null : draft.modelName.trim(),
      provider: draft.provider.trim() === "" ? null : draft.provider.trim(),
      image_model:
        draft.imageModel.trim() === "" ? null : draft.imageModel.trim(),
      max_tokens: asInt(draft.maxTokens, 0) || null,
      context_window: asInt(draft.contextWindow, 0) || null,
      temperature:
        draft.temperature.trim() === "" ? null : asFloat(draft.temperature, 0),
      max_tool_iterations: asInt(draft.maxToolIterations, 0) || null,
    },
    list: draft.agents.map((a) => ({
      id: a.id.trim(),
      name: a.name.trim() === "" ? null : a.name.trim(),
      default: a.isDefault,
      model: a.model.trim() === "" ? null : a.model.trim(),
      skills: a.skills
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s !== ""),
    })),
    dispatch: parseDispatchRules(draft.dispatchJson),
  }
}

export function AdvancedConfigPage() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  const [contextManager, setContextManager] = useState("")
  const [semantic, setSemantic] = useState<SemanticMemoryConfig>({
    ...DEFAULT_SEMANTIC,
  })
  const [imageModel, setImageModel] = useState("")
  const [imageModelFallbacksText, setImageModelFallbacksText] = useState("")
  const [routingEnabled, setRoutingEnabled] = useState(false)
  const [routingLightModel, setRoutingLightModel] = useState("")
  const [routingThreshold, setRoutingThreshold] = useState("0.6")
  const [baseline, setBaseline] = useState<string>("")
  const [treeSel, setTreeSel] = useState<string>("live")
  const [treeDraft, setTreeDraft] = useState<AgentTreeDraft>({ ...EMPTY_TREE })
  const [treeBaseline, setTreeBaseline] = useState("")
  const [treeLoadedFor, setTreeLoadedFor] = useState<string>("")
  const [newProfileName, setNewProfileName] = useState("")

  const { data, isLoading, error } = useQuery({
    queryKey: ["config"],
    queryFn: async () => {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 5000)
      try {
        const res = await launcherFetch("/api/config", {
          signal: controller.signal,
        })
        if (!res.ok) {
          throw new Error("Failed to load config")
        }
        return res.json()
      } finally {
        clearTimeout(timer)
      }
    },
  })

  const loaded = Boolean(data)
  if (data && baseline === "") {
    const defaults = (data as Record<string, any>).agents?.defaults ?? {}
    setBaseline(JSON.stringify(defaults))
    setContextManager(
      typeof defaults.context_manager === "string"
        ? defaults.context_manager
        : "",
    )
    setSemantic(readSemantic(defaults.context_manager_config))
    setImageModel(
      typeof defaults.image_model === "string" ? defaults.image_model : "",
    )
    setImageModelFallbacksText(
      Array.isArray(defaults.image_model_fallbacks)
        ? (defaults.image_model_fallbacks as string[]).join("\n")
        : "",
    )
    const routing = defaults.routing ?? {}
    setRoutingEnabled(Boolean(routing.enabled ?? false))
    setRoutingLightModel(
      typeof routing.light_model === "string" ? routing.light_model : "",
    )
    setRoutingThreshold(String(routing.threshold ?? 0.6))
  }

  const agentsData = (data as Record<string, any>)?.agents ?? {}
  const profiles = (agentsData.profiles ?? {}) as Record<string, any>
  const profileNames = Object.keys(profiles).sort()
  const activeProfile =
    typeof agentsData.active_profile === "string"
      ? agentsData.active_profile
      : ""

  if (data && treeLoadedFor !== treeSel) {
    const raw = treeSel === "live" ? agentsData : profiles[treeSel]
    if (raw) {
      const draft = treeFromRaw(raw)
      setTreeDraft(draft)
      setTreeBaseline(JSON.stringify(draft))
      setTreeLoadedFor(treeSel)
    }
  }

  const treeDirty =
    treeBaseline !== "" && JSON.stringify(treeDraft) !== treeBaseline

  const dirty =
    baseline !== "" &&
    JSON.stringify({
      context_manager: contextManager,
      semantic,
      image_model: imageModel,
      image_model_fallbacks: imageModelFallbacksText,
      routing: {
        enabled: routingEnabled,
        light_model: routingLightModel,
        threshold: routingThreshold,
      },
    }) !==
      JSON.stringify(
        (() => {
          const defaults = (data as Record<string, any>)?.agents?.defaults ?? {}
          const routing = defaults.routing ?? {}
          return {
            context_manager: defaults.context_manager ?? "",
            semantic: readSemantic(defaults.context_manager_config),
            image_model: defaults.image_model ?? "",
            image_model_fallbacks: Array.isArray(defaults.image_model_fallbacks)
              ? (defaults.image_model_fallbacks as string[]).join("\n")
              : "",
            routing: {
              enabled: Boolean(routing.enabled ?? false),
              light_model: routing.light_model ?? "",
              threshold: String(routing.threshold ?? 0.6),
            },
          }
        })(),
      )

  const mutation = useMutation({
    mutationFn: async (patch: Record<string, unknown>) => {
      const res = await launcherFetch("/api/config", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || "Failed to save config")
      }
      return res.json()
    },
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: ["config"] })
      const gateway = await refreshGatewayState({ force: true })
      showSaveSuccessOrRestartToast(
        t,
        t("pages.config_advanced.saved"),
        t("navigation.advanced_settings"),
        gateway?.restartRequired === true,
      )
    },
    onError: (err: Error) => {
      toast.error(err.message || t("pages.config_advanced.save_failed"))
    },
  })

  function handleSave() {
    const fallbacks = imageModelFallbacksText
      .split("\n")
      .map((s) => s.trim())
      .filter((s) => s !== "")

    const patch: Record<string, unknown> = {
      agents: {
        defaults: {
          context_manager: contextManager,
          context_manager_config: {
            enableSemantic: semantic.enableSemantic,
            embeddingEndpoint: semantic.embeddingEndpoint.trim(),
            embeddingModel: semantic.embeddingModel.trim(),
            embeddingDim: asInt(String(semantic.embeddingDim), 1024),
            topK: asInt(String(semantic.topK), 8),
            minScore: asFloat(String(semantic.minScore), 0.35),
            asyncWrite: semantic.asyncWrite,
          },
          image_model: imageModel.trim() === "" ? null : imageModel.trim(),
          image_model_fallbacks:
            fallbacks.length > 0 ? fallbacks : null,
          routing: {
            enabled: routingEnabled,
            light_model:
              routingLightModel.trim() === "" ? null : routingLightModel.trim(),
            threshold: asFloat(routingThreshold, 0.6),
          },
        },
      },
    }
    mutation.mutate(patch)
  }

  function handleReset() {
    if (data) {
      const defaults = (data as Record<string, any>).agents?.defaults ?? {}
      setContextManager(
        typeof defaults.context_manager === "string"
          ? defaults.context_manager
          : "",
      )
      setSemantic(readSemantic(defaults.context_manager_config))
      setImageModel(
        typeof defaults.image_model === "string" ? defaults.image_model : "",
      )
      setImageModelFallbacksText(
        Array.isArray(defaults.image_model_fallbacks)
          ? (defaults.image_model_fallbacks as string[]).join("\n")
          : "",
      )
      const routing = defaults.routing ?? {}
      setRoutingEnabled(Boolean(routing.enabled ?? false))
      setRoutingLightModel(
        typeof routing.light_model === "string" ? routing.light_model : "",
      )
      setRoutingThreshold(String(routing.threshold ?? 0.6))
      setBaseline(JSON.stringify(defaults))
    }
  }

  const treeApplyMutation = useMutation({
    mutationFn: async (profile: string) => {
      const res = await launcherFetch("/api/agents/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile }),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || "Failed to apply profile")
      }
      return res.json()
    },
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: ["config"] })
      await refreshGatewayState({ force: true })
      toast.success(t("pages.config_advanced.tree_apply_success"))
    },
    onError: (err: Error) => {
      toast.error(err.message || t("pages.config_advanced.tree_apply_failed"))
    },
  })

  const treeSaveProfileMutation = useMutation({
    mutationFn: async (payload: {
      name: string
      tree: Record<string, unknown>
    }) => {
      const res = await launcherFetch("/api/agents/profile", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || "Failed to save profile")
      }
      return res.json()
    },
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: ["config"] })
      await refreshGatewayState({ force: true })
      toast.success(t("pages.config_advanced.tree_saved"))
    },
    onError: (err: Error) => {
      toast.error(err.message || t("pages.config_advanced.tree_save_failed"))
    },
  })

  const treeDeleteMutation = useMutation({
    mutationFn: async (profile: string) => {
      const res = await launcherFetch("/api/agents/profile", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: profile }),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || "Failed to delete profile")
      }
      return res.json()
    },
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: ["config"] })
      await refreshGatewayState({ force: true })
      toast.success(t("pages.config_advanced.tree_deleted"))
    },
    onError: (err: Error) => {
      toast.error(err.message || t("pages.config_advanced.tree_delete_failed"))
    },
  })

  const treeSaving =
    treeApplyMutation.isPending ||
    treeSaveProfileMutation.isPending ||
    treeDeleteMutation.isPending ||
    mutation.isPending

  function handleTreeSave() {
    if (treeDraft.agents.some((a) => a.id.trim() === "")) {
      toast.error(t("pages.config_advanced.tree_agent_id_required"))
      return
    }
    let patch: Record<string, unknown>
    try {
      patch = treeToPatch(treeDraft)
    } catch (e) {
      toast.error(
        (e as Error).message || t("pages.config_advanced.tree_dispatch_invalid"),
      )
      return
    }
    if (treeSel === "live") {
      // Live tree: description lives on profiles only.
      const agentsPatch: Record<string, unknown> = { ...patch }
      delete agentsPatch.description
      mutation.mutate({ agents: agentsPatch })
    } else {
      treeSaveProfileMutation.mutate({ name: treeSel, tree: patch })
    }
  }

  function handleTreeApply() {
    if (treeSel === "live" || treeSel === activeProfile) return
    treeApplyMutation.mutate(treeSel)
  }

  function handleTreeDelete() {
    if (treeSel === "live") return
    if (!window.confirm(t("pages.config_advanced.tree_delete_confirm"))) return
    treeDeleteMutation.mutate(treeSel, {
      onSuccess: () => {
        setTreeSel("live")
        setTreeLoadedFor("")
      },
    })
  }

  function handleTreeNew() {
    const name = newProfileName.trim()
    if (!name) return
    let patch: Record<string, unknown>
    try {
      patch = treeToPatch(treeDraft)
    } catch (e) {
      toast.error(
        (e as Error).message || t("pages.config_advanced.tree_dispatch_invalid"),
      )
      return
    }
    setTreeSel(name)
    setNewProfileName("")
    treeSaveProfileMutation.mutate({ name, tree: patch })
  }

  function updateAgentRow(
    index: number,
    updater: (row: AgentRowDraft) => AgentRowDraft,
  ) {
    setTreeDraft((d) => ({
      ...d,
      agents: d.agents.map((row, i) => (i === index ? updater(row) : row)),
    }))
  }

  return (
    <div className="space-y-6">
      <PageHeader title={t("pages.config_advanced.title")}>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={handleReset}
            disabled={!dirty || mutation.isPending}
          >
            {t("common.reset")}
          </Button>
          <Button
            onClick={handleSave}
            disabled={!dirty || mutation.isPending}
          >
            {mutation.isPending
              ? t("common.saving")
              : t("common.save")}
          </Button>
        </div>
      </PageHeader>

      <ConfigChangeNotice
        kind="save"
        title={t("common.saveChangesTitle")}
        description={t("pages.config.unsaved_changes")}
      />

      {isLoading && (
        <Card size="sm">
          <CardContent className="py-8 text-center text-muted-foreground">
            {t("pages.config_advanced.loading")}
          </CardContent>
        </Card>
      )}
      {error && (
        <Card size="sm">
          <CardContent className="py-8 text-center text-destructive">
            {t("pages.config.load_error")}
          </CardContent>
        </Card>
      )}
      {loaded && !error && (
        <div className="space-y-6">
          {/* Agent Trees & Profiles */}
          <Card size="sm">
            <CardHeader className="border-border border-b">
              <CardTitle>{t("pages.config_advanced.trees_title")}</CardTitle>
              <CardDescription>
                {t("pages.config_advanced.trees_desc")}
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="divide-border/70 divide-y">
                <Field
                  label={t("pages.config_advanced.tree_select")}
                  hint={t("pages.config_advanced.tree_select_hint")}
                  layout="setting-row"
                >
                  <div className="flex w-full items-center gap-2">
                    <Select value={treeSel} onValueChange={setTreeSel}>
                      <SelectTrigger className="h-9">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="live">
                          {t("pages.config_advanced.tree_live")}
                        </SelectItem>
                        {profileNames.map((p) => (
                          <SelectItem key={p} value={p}>
                            {p}
                            {p === activeProfile
                              ? ` (${t("pages.config_advanced.tree_active")})`
                              : ""}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {treeSel !== "live" && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={handleTreeApply}
                        disabled={
                          treeSel === activeProfile || treeSaving
                        }
                      >
                        {t("pages.config_advanced.tree_apply")}
                      </Button>
                    )}
                  </div>
                </Field>

                <Field
                  label={t("pages.config_advanced.tree_description")}
                  hint={t("pages.config_advanced.tree_description_hint")}
                  layout="setting-row"
                >
                  <Input
                    className="h-9"
                    value={treeDraft.description}
                    onChange={(e) =>
                      setTreeDraft((d) => ({
                        ...d,
                        description: e.target.value,
                      }))
                    }
                    placeholder={t("pages.config_advanced.tree_description_placeholder")}
                  />
                </Field>

                <div className="py-3">
                  <p className="text-foreground mb-2 text-sm font-medium">
                    {t("pages.config_advanced.tree_defaults_title")}
                  </p>
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    <Field
                      label={t("pages.config_advanced.model_name")}
                      layout="default"
                    >
                      <Input
                        className="h-9"
                        value={treeDraft.modelName}
                        onChange={(e) =>
                          setTreeDraft((d) => ({
                            ...d,
                            modelName: e.target.value,
                          }))
                        }
                        placeholder="openrouter-flash-byok"
                      />
                    </Field>
                    <Field
                      label={t("pages.config_advanced.provider")}
                      layout="default"
                    >
                      <Input
                        className="h-9"
                        value={treeDraft.provider}
                        onChange={(e) =>
                          setTreeDraft((d) => ({
                            ...d,
                            provider: e.target.value,
                          }))
                        }
                        placeholder="openrouter"
                      />
                    </Field>
                    <Field
                      label={t("pages.config_advanced.image_model")}
                      layout="default"
                    >
                      <Input
                        className="h-9"
                        value={treeDraft.imageModel}
                        onChange={(e) =>
                          setTreeDraft((d) => ({
                            ...d,
                            imageModel: e.target.value,
                          }))
                        }
                        placeholder="skynet-qwen3-vl"
                      />
                    </Field>
                    <Field
                      label={t("pages.config_advanced.max_tokens")}
                      layout="default"
                    >
                      <Input
                        className="h-9"
                        type="number"
                        value={treeDraft.maxTokens}
                        onChange={(e) =>
                          setTreeDraft((d) => ({
                            ...d,
                            maxTokens: e.target.value,
                          }))
                        }
                      />
                    </Field>
                    <Field
                      label={t("pages.config_advanced.context_window")}
                      layout="default"
                    >
                      <Input
                        className="h-9"
                        type="number"
                        value={treeDraft.contextWindow}
                        onChange={(e) =>
                          setTreeDraft((d) => ({
                            ...d,
                            contextWindow: e.target.value,
                          }))
                        }
                      />
                    </Field>
                    <Field
                      label={t("pages.config_advanced.temperature")}
                      layout="default"
                    >
                      <Input
                        className="h-9"
                        type="number"
                        step="0.1"
                        value={treeDraft.temperature}
                        onChange={(e) =>
                          setTreeDraft((d) => ({
                            ...d,
                            temperature: e.target.value,
                          }))
                        }
                      />
                    </Field>
                    <Field
                      label={t("pages.config_advanced.max_tool_iterations")}
                      layout="default"
                    >
                      <Input
                        className="h-9"
                        type="number"
                        value={treeDraft.maxToolIterations}
                        onChange={(e) =>
                          setTreeDraft((d) => ({
                            ...d,
                            maxToolIterations: e.target.value,
                          }))
                        }
                      />
                    </Field>
                  </div>
                </div>

                <div className="py-3">
                  <div className="mb-2 flex items-center justify-between">
                    <p className="text-foreground text-sm font-medium">
                      {t("pages.config_advanced.tree_agents_title")}
                    </p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        setTreeDraft((d) => ({
                          ...d,
                          agents: [
                            ...d.agents,
                            {
                              id: "",
                              name: "",
                              model: "",
                              skills: "",
                              isDefault: false,
                            },
                          ],
                        }))
                      }
                    >
                      {t("pages.config_advanced.tree_add_agent")}
                    </Button>
                  </div>
                  <div className="space-y-2">
                    {treeDraft.agents.length === 0 && (
                      <p className="text-muted-foreground text-xs">
                        {t("pages.config_advanced.tree_no_agents")}
                      </p>
                    )}
                    {treeDraft.agents.map((row, idx) => (
                      <div
                        key={idx}
                        className="border-border/70 bg-muted/30 grid grid-cols-1 items-end gap-2 rounded-md border p-2 md:grid-cols-[1fr_1fr_1fr_1fr_auto_auto]"
                      >
                        <Field
                          label={t("pages.config_advanced.agent_id")}
                          layout="default"
                        >
                          <Input
                            className="h-9"
                            value={row.id}
                            onChange={(e) =>
                              updateAgentRow(idx, (r) => ({
                                ...r,
                                id: e.target.value,
                              }))
                            }
                            placeholder="main"
                          />
                        </Field>
                        <Field
                          label={t("pages.config_advanced.agent_name")}
                          layout="default"
                        >
                          <Input
                            className="h-9"
                            value={row.name}
                            onChange={(e) =>
                              updateAgentRow(idx, (r) => ({
                                ...r,
                                name: e.target.value,
                              }))
                            }
                          />
                        </Field>
                        <Field
                          label={t("pages.config_advanced.agent_model")}
                          layout="default"
                        >
                          <Input
                            className="h-9"
                            value={row.model}
                            onChange={(e) =>
                              updateAgentRow(idx, (r) => ({
                                ...r,
                                model: e.target.value,
                              }))
                            }
                            placeholder="openrouter-flash-byok"
                          />
                        </Field>
                        <Field
                          label={t("pages.config_advanced.agent_skills")}
                          layout="default"
                        >
                          <Input
                            className="h-9"
                            value={row.skills}
                            onChange={(e) =>
                              updateAgentRow(idx, (r) => ({
                                ...r,
                                skills: e.target.value,
                              }))
                            }
                            placeholder="summarize, weather"
                          />
                        </Field>
                        <label className="flex h-9 cursor-pointer items-center gap-1.5 text-sm">
                          <input
                            type="checkbox"
                            checked={row.isDefault}
                            onChange={(e) =>
                              updateAgentRow(idx, (r) => ({
                                ...r,
                                isDefault: e.target.checked,
                              }))
                            }
                          />
                          {t("pages.config_advanced.agent_default")}
                        </label>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() =>
                            setTreeDraft((d) => ({
                              ...d,
                              agents: d.agents.filter((_, i) => i !== idx),
                            }))
                          }
                        >
                          {t("pages.config_advanced.tree_remove_agent")}
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>

                <Field
                  label={t("pages.config_advanced.tree_dispatch")}
                  hint={t("pages.config_advanced.tree_dispatch_hint")}
                  layout="setting-row"
                  controlClassName="md:max-w-[28rem]"
                >
                  <Textarea
                    className="font-mono min-h-[90px] text-xs"
                    value={treeDraft.dispatchJson}
                    onChange={(e) =>
                      setTreeDraft((d) => ({
                        ...d,
                        dispatchJson: e.target.value,
                      }))
                    }
                  />
                </Field>

                <div className="flex flex-wrap items-center gap-2 py-3">
                  <Button
                    onClick={handleTreeSave}
                    disabled={!treeDirty || treeSaving}
                  >
                    {treeSaving
                      ? t("common.saving")
                      : t("pages.config_advanced.tree_save")}
                  </Button>
                  {treeSel !== "live" && (
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleTreeDelete}
                      disabled={treeSaving}
                    >
                      {t("pages.config_advanced.tree_delete")}
                    </Button>
                  )}
                  <div className="ml-auto flex items-center gap-2">
                    <Input
                      className="h-9 w-44"
                      value={newProfileName}
                      onChange={(e) => setNewProfileName(e.target.value)}
                      placeholder={t("pages.config_advanced.tree_new_placeholder")}
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleTreeNew}
                      disabled={!newProfileName.trim() || treeSaving}
                    >
                      {t("pages.config_advanced.tree_new")}
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Context manager */}
          <Card size="sm">
            <CardHeader className="border-border border-b">
              <CardTitle>
                {t("pages.config_advanced.context_manager_title")}
              </CardTitle>
              <CardDescription>
                {t("pages.config_advanced.context_manager_desc")}
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="divide-border/70 divide-y">
                <Field
                  label={t("pages.config_advanced.context_manager")}
                  hint={t("pages.config_advanced.context_manager_hint")}
                  layout="setting-row"
                >
                  <Select
                    value={contextManager === "" ? "__default__" : contextManager}
                    onValueChange={(v) =>
                      setContextManager(v === "__default__" ? "" : v)
                    }
                  >
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__default__">
                        {t("pages.config_advanced.context_manager_default")}
                      </SelectItem>
                      <SelectItem value="seahorse">
                        {t("pages.config_advanced.context_manager_seahorse")}
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
              </div>
            </CardContent>
          </Card>

          {/* Semantic memory */}
          <Card size="sm">
            <CardHeader className="border-border border-b">
              <CardTitle>
                {t("pages.config_advanced.semantic_title")}
              </CardTitle>
              <CardDescription>
                {t("pages.config_advanced.semantic_desc")}
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="divide-border/70 divide-y">
                <SwitchCardField
                  label={t("pages.config_advanced.semantic_enable")}
                  hint={t("pages.config_advanced.semantic_enable_hint")}
                  checked={semantic.enableSemantic}
                  onCheckedChange={(v) =>
                    setSemantic((s) => ({ ...s, enableSemantic: v }))
                  }
                />
                <Field
                  label={t("pages.config_advanced.embedding_endpoint")}
                  hint={t("pages.config_advanced.embedding_endpoint_hint")}
                  layout="setting-row"
                >
                  <Input
                    className="h-9"
                    value={semantic.embeddingEndpoint}
                    onChange={(e) =>
                      setSemantic((s) => ({
                        ...s,
                        embeddingEndpoint: e.target.value,
                      }))
                    }
                    placeholder="http://host:18084/v1/embeddings"
                  />
                </Field>
                <Field
                  label={t("pages.config_advanced.embedding_model")}
                  hint={t("pages.config_advanced.embedding_model_hint")}
                  layout="setting-row"
                >
                  <Input
                    className="h-9"
                    value={semantic.embeddingModel}
                    onChange={(e) =>
                      setSemantic((s) => ({
                        ...s,
                        embeddingModel: e.target.value,
                      }))
                    }
                    placeholder="jina-embeddings-v3"
                  />
                </Field>
                <Field
                  label={t("pages.config_advanced.embedding_dim")}
                  hint={t("pages.config_advanced.embedding_dim_hint")}
                  layout="setting-row"
                >
                  <Input
                    className="h-9"
                    type="number"
                    value={String(semantic.embeddingDim)}
                    onChange={(e) =>
                      setSemantic((s) => ({
                        ...s,
                        embeddingDim: asInt(e.target.value, 1024),
                      }))
                    }
                  />
                </Field>
                <Field
                  label={t("pages.config_advanced.semantic_topk")}
                  hint={t("pages.config_advanced.semantic_topk_hint")}
                  layout="setting-row"
                >
                  <Input
                    className="h-9"
                    type="number"
                    value={String(semantic.topK)}
                    onChange={(e) =>
                      setSemantic((s) => ({
                        ...s,
                        topK: asInt(e.target.value, 8),
                      }))
                    }
                  />
                </Field>
                <Field
                  label={t("pages.config_advanced.semantic_min_score")}
                  hint={t("pages.config_advanced.semantic_min_score_hint")}
                  layout="setting-row"
                >
                  <Input
                    className="h-9"
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    value={String(semantic.minScore)}
                    onChange={(e) =>
                      setSemantic((s) => ({
                        ...s,
                        minScore: asFloat(e.target.value, 0.35),
                      }))
                    }
                  />
                </Field>
                <SwitchCardField
                  label={t("pages.config_advanced.semantic_async")}
                  hint={t("pages.config_advanced.semantic_async_hint")}
                  checked={semantic.asyncWrite}
                  onCheckedChange={(v) =>
                    setSemantic((s) => ({ ...s, asyncWrite: v }))
                  }
                />
              </div>
            </CardContent>
          </Card>

          {/* Vision */}
          <Card size="sm">
            <CardHeader className="border-border border-b">
              <CardTitle>{t("pages.config_advanced.vision_title")}</CardTitle>
              <CardDescription>
                {t("pages.config_advanced.vision_desc")}
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="divide-border/70 divide-y">
                <Field
                  label={t("pages.config_advanced.image_model")}
                  hint={t("pages.config_advanced.image_model_hint")}
                  layout="setting-row"
                >
                  <Input
                    className="h-9"
                    value={imageModel}
                    onChange={(e) => setImageModel(e.target.value)}
                    placeholder="skynet-qwen3-vl"
                  />
                </Field>
                <Field
                  label={t("pages.config_advanced.image_model_fallbacks")}
                  hint={t("pages.config_advanced.image_model_fallbacks_hint")}
                  layout="setting-row"
                  controlClassName="md:max-w-[28rem]"
                >
                  <Textarea
                    className="min-h-[80px]"
                    value={imageModelFallbacksText}
                    onChange={(e) => setImageModelFallbacksText(e.target.value)}
                    placeholder={"gpt-5.6-luna\nskynet-qwen-vl"}
                  />
                </Field>
              </div>
            </CardContent>
          </Card>

          {/* Routing */}
          <Card size="sm">
            <CardHeader className="border-border border-b">
              <CardTitle>{t("pages.config_advanced.routing_title")}</CardTitle>
              <CardDescription>
                {t("pages.config_advanced.routing_desc")}
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="divide-border/70 divide-y">
                <SwitchCardField
                  label={t("pages.config_advanced.routing_enable")}
                  hint={t("pages.config_advanced.routing_enable_hint")}
                  checked={routingEnabled}
                  onCheckedChange={setRoutingEnabled}
                />
                <Field
                  label={t("pages.config_advanced.routing_light_model")}
                  hint={t("pages.config_advanced.routing_light_model_hint")}
                  layout="setting-row"
                >
                  <Input
                    className="h-9"
                    value={routingLightModel}
                    onChange={(e) => setRoutingLightModel(e.target.value)}
                    placeholder="openrouter-flash-byok"
                  />
                </Field>
                <Field
                  label={t("pages.config_advanced.routing_threshold")}
                  hint={t("pages.config_advanced.routing_threshold_hint")}
                  layout="setting-row"
                >
                  <Input
                    className="h-9"
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    value={routingThreshold}
                    onChange={(e) => setRoutingThreshold(e.target.value)}
                  />
                </Field>
              </div>
            </CardContent>
          </Card>

          <p className="text-muted-foreground text-xs">
            {t("pages.config_advanced.advanced_notice")}
          </p>
        </div>
      )}
    </div>
  )
}
