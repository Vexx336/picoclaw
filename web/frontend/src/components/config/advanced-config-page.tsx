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
                    value={contextManager}
                    onValueChange={setContextManager}
                  >
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">
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
