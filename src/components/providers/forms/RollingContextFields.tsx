import { useTranslation } from "react-i18next";
import {
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormDescription,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { UseFormReturn } from "react-hook-form";
import { ProviderFormData } from "@/lib/schemas/provider";

interface RollingContextFieldsProps {
  form: UseFormReturn<ProviderFormData>;
}

export function RollingContextFields({ form }: RollingContextFieldsProps) {
  const { t } = useTranslation();
  const enabled = form.watch("rollingContextEnabled");
  const rollingEnabled = form.watch("rollingContextEnabled");
  const nativeEnabled = form.watch("nativeAutoCompactEnabled");

  return (
    <div className="space-y-4 border rounded-lg p-4">
      <h3 className="font-medium text-sm">
        {t("provider.form.rollingContext.title")}
      </h3>

      <FormField
        control={form.control}
        name="rollingContextEnabled"
        render={({ field }) => (
          <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3 shadow-sm">
            <div className="space-y-0.5">
              <FormLabel>{t("provider.form.rollingContext.enabled")}</FormLabel>
              <FormDescription>
                {t("provider.form.rollingContext.enabledHint")}
              </FormDescription>
            </div>
            <FormControl>
              <Switch
                checked={field.value ?? false}
                onCheckedChange={field.onChange}
              />
            </FormControl>
          </FormItem>
        )}
      />

      {enabled && (
        <>
          <FormField
            control={form.control}
            name="contextWindow"
            render={({ field }) => (
              <FormItem>
                <FormLabel>{t("provider.form.contextWindow")}</FormLabel>
                <FormControl>
                  <Input
                    type="number"
                    placeholder={t("provider.form.contextWindowPlaceholder")}
                    value={field.value ?? ""}
                    onChange={(e) =>
                      field.onChange(
                        e.target.value === ""
                          ? undefined
                          : parseInt(e.target.value, 10)
                      )
                    }
                  />
                </FormControl>
                <FormDescription>
                  {t("provider.form.rollingContext.contextWindowHint")}
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="rollingContextThreshold"
            render={({ field }) => (
              <FormItem>
                <FormLabel>{t("provider.form.rollingContext.threshold")}</FormLabel>
                <FormControl>
                  <Input
                    type="number"
                    step="0.05"
                    min="0.1"
                    max="0.99"
                    placeholder="0.8"
                    value={field.value ?? ""}
                    onChange={(e) =>
                      field.onChange(
                        e.target.value === ""
                          ? undefined
                          : parseFloat(e.target.value)
                      )
                    }
                  />
                </FormControl>
                <FormDescription>
                  {t("provider.form.rollingContext.thresholdHint")}
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="rollingContextPreserveRounds"
            render={({ field }) => (
              <FormItem>
                <FormLabel>{t("provider.form.rollingContext.preserveRounds")}</FormLabel>
                <FormControl>
                  <Input
                    type="number"
                    min="1"
                    placeholder="6"
                    value={field.value ?? ""}
                    onChange={(e) =>
                      field.onChange(
                        e.target.value === ""
                          ? undefined
                          : parseInt(e.target.value, 10)
                      )
                    }
                  />
                </FormControl>
                <FormDescription>
                  {t("provider.form.rollingContext.preserveRoundsHint")}
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
        </>
      )}
      {/* ---- Section 1: rolling-context (proxy mode) ---- */}
      <div className="space-y-4 pl-2 border-l-2 border-blue-500/30">
        <p className="text-xs text-muted-foreground">
          {t("provider.form.rollingContext.proxyModeHint")}
        </p>

        <FormField
          control={form.control}
          name="rollingContextEnabled"
          render={({ field }) => (
            <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3 shadow-sm">
              <div className="space-y-0.5">
                <FormLabel>
                  {t("provider.form.rollingContext.enabled")}
                </FormLabel>
                <FormDescription>
                  {t("provider.form.rollingContext.enabledHint")}
                </FormDescription>
              </div>
              <FormControl>
                <Switch
                  checked={field.value ?? false}
                  onCheckedChange={field.onChange}
                />
              </FormControl>
            </FormItem>
          )}
        />

        {rollingEnabled && (
          <>
            <FormField
              control={form.control}
              name="contextWindow"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>
                    {t("provider.form.contextWindow")}
                  </FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      placeholder={t("provider.form.contextWindowPlaceholder")}
                      value={field.value ?? ""}
                      onChange={(e) =>
                        field.onChange(
                          e.target.value === ""
                            ? undefined
                            : parseInt(e.target.value, 10)
                        )
                      }
                    />
                  </FormControl>
                  <FormDescription>
                    {t("provider.form.rollingContext.contextWindowHint")}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="rollingContextThreshold"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>
                    {t("provider.form.rollingContext.threshold")}
                  </FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      step="0.05"
                      min="0.1"
                      max="0.99"
                      placeholder="0.8"
                      value={field.value ?? ""}
                      onChange={(e) =>
                        field.onChange(
                          e.target.value === ""
                            ? undefined
                            : parseFloat(e.target.value)
                        )
                      }
                    />
                  </FormControl>
                  <FormDescription>
                    {t("provider.form.rollingContext.thresholdHint")}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="rollingContextPreserveRounds"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>
                    {t("provider.form.rollingContext.preserveRounds")}
                  </FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      min="1"
                      placeholder="6"
                      value={field.value ?? ""}
                      onChange={(e) =>
                        field.onChange(
                          e.target.value === ""
                            ? undefined
                            : parseInt(e.target.value, 10)
                        )
                      }
                    />
                  </FormControl>
                  <FormDescription>
                    {t("provider.form.rollingContext.preserveRoundsHint")}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
          </>
        )}
      </div>

      {/* ---- Section 2: native auto-compact (direct mode fallback) ---- */}
      <div className="space-y-4 pl-2 border-l-2 border-amber-500/30">
        <p className="text-xs text-muted-foreground">
          {t("provider.form.rollingContext.nativeModeHint")}
        </p>

        <FormField
          control={form.control}
          name="nativeAutoCompactEnabled"
          render={({ field }) => (
            <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3 shadow-sm">
              <div className="space-y-0.5">
                <FormLabel>
                  {t("provider.form.rollingContext.nativeEnabled")}
                </FormLabel>
                <FormDescription>
                  {t("provider.form.rollingContext.nativeEnabledHint")}
                </FormDescription>
              </div>
              <FormControl>
                <Switch
                  checked={field.value ?? false}
                  onCheckedChange={field.onChange}
                />
              </FormControl>
            </FormItem>
          )}
        />

        {nativeEnabled && (
          <div className="space-y-3 rounded-lg border p-3 bg-amber-50/30 dark:bg-amber-950/20">
            <p className="text-xs text-amber-700 dark:text-amber-300">
              {t("provider.form.rollingContext.nativeConfigHint")}
            </p>

            <FormField
              control={form.control}
              name="nativeAutoCompactPct"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>
                    {t("provider.form.rollingContext.nativePct")}
                  </FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      min="10"
                      max="99"
                      step="5"
                      placeholder="60"
                      value={field.value ?? ""}
                      onChange={(e) =>
                        field.onChange(
                          e.target.value === ""
                            ? undefined
                            : parseInt(e.target.value, 10)
                        )
                      }
                    />
                  </FormControl>
                  <FormDescription>
                    {t("provider.form.rollingContext.nativePctHint")}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="nativeAutoCompactWindow"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>
                    {t("provider.form.rollingContext.nativeWindow")}
                  </FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      placeholder={
                        form.getValues("contextWindow")?.toString() ??
                        "1000000"
                      }
                      value={field.value ?? ""}
                      onChange={(e) =>
                        field.onChange(
                          e.target.value === ""
                            ? undefined
                            : parseInt(e.target.value, 10)
                        )
                      }
                    />
                  </FormControl>
                  <FormDescription>
                    {t("provider.form.rollingContext.nativeWindowHint")}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Show the env vars that will be written */}
            <div className="rounded border border-amber-300/50 bg-amber-100/50 dark:bg-amber-950/40 p-2 text-xs font-mono text-amber-900 dark:text-amber-200">
              <div>
                CLAUDE_AUTOCOMPACT_PCT_OVERRIDE ={" "}
                {form.watch("nativeAutoCompactPct") ?? 60}
              </div>
              <div>
                CLAUDE_CODE_AUTO_COMPACT_WINDOW ={" "}
                {form.watch("nativeAutoCompactWindow") ??
                  form.watch("contextWindow") ??
                  1000000}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
