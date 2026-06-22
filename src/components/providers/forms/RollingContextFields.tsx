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
import { UseFormReturn } from "react-hook-form";
import { ProviderFormData } from "@/lib/schemas/provider";

interface RollingContextFieldsProps {
  form: UseFormReturn<ProviderFormData>;
}

export function RollingContextFields({ form }: RollingContextFieldsProps) {
  const { t } = useTranslation();

  return (
    <div className="space-y-4 border rounded-lg p-4">
      <h3 className="font-medium text-sm">
        {t("provider.form.rollingContext.title")}
      </h3>
      <p className="text-xs text-muted-foreground">
        {t("provider.form.rollingContext.overviewHint")}
      </p>

      <FormField
        control={form.control}
        name="contextWindow"
        render={({ field }) => (
          <FormItem>
            <FormLabel>
              {t("provider.form.rollingContext.contextWindowLabel")}
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
        name="nativeAutoCompactPct"
        render={({ field }) => (
          <FormItem>
            <FormLabel>
              {t("provider.form.rollingContext.compactPctLabel")}
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
              {t("provider.form.rollingContext.compactPctHint")}
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />

      <div className="rounded border p-2 text-xs font-mono bg-muted/50">
        <div>
          CLAUDE_AUTOCOMPACT_PCT_OVERRIDE ={" "}
          {form.watch("nativeAutoCompactPct") ?? 60}
        </div>
        <div>
          CLAUDE_CODE_AUTO_COMPACT_WINDOW ={" "}
          {form.watch("contextWindow") ?? 1000000}
        </div>
      </div>
    </div>
  );
}
