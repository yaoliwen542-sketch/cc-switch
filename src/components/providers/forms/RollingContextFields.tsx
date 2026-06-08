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
    </div>
  );
}
