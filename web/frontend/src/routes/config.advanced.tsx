import { createFileRoute } from "@tanstack/react-router"

import { AdvancedConfigPage } from "@/components/config/advanced-config-page"

export const Route = createFileRoute("/config/advanced")({
  component: AdvancedConfigPage,
})
