//! 一次性迁移：把 per-provider 的 rolling context 开关/策略迁移到全局 proxy rolling context。
//!
//! 迁移只读 provider.meta，不写回 provider；老字段保留以保证向后兼容。

use crate::database::Database;
use crate::settings::{
    is_proxy_rolling_context_migrated, mutate_settings, ProxyRollingContextMigration,
};
use crate::{app_config::AppType, error::AppError};

/// 迁移结果。
pub struct MigrationOutcome {
    pub skipped_reason: Option<String>,
    pub had_enabled_provider: bool,
    pub preserve_rounds: u32,
    pub target: f64,
}

impl MigrationOutcome {
    fn skipped(reason: impl Into<String>) -> Self {
        Self {
            skipped_reason: Some(reason.into()),
            had_enabled_provider: false,
            preserve_rounds: 6,
            target: 0.6,
        }
    }
}

/// 扫描所有供应商，若发现任何供应商启用了 rolling context，则打开全局开关，
/// 并把第一个供应商的 preserve_rounds / target 复制到全局设置。
pub fn maybe_migrate_provider_rolling_context_to_global(
    db: &Database,
) -> Result<MigrationOutcome, AppError> {
    if is_proxy_rolling_context_migrated() {
        return Ok(MigrationOutcome::skipped("already_migrated"));
    }

    let app_types = [
        AppType::Claude,
        AppType::ClaudeDesktop,
        AppType::Codex,
        AppType::Gemini,
        AppType::OpenCode,
        AppType::OpenClaw,
        AppType::Hermes,
    ];

    let mut had_enabled_provider = false;
    let mut preserve_rounds: Option<u32> = None;
    let mut target: Option<f64> = None;

    for app_type in &app_types {
        let providers = db.get_all_providers(app_type.as_str());
        let providers = match providers {
            Ok(p) => p,
            Err(e) => {
                log::warn!(
                    "扫描 {} 供应商失败，跳过该应用类型: {}",
                    app_type.as_str(),
                    e
                );
                continue;
            }
        };

        for (_id, provider) in providers {
            let Some(meta) = provider.meta else { continue };
            if meta.rolling_context_enabled.unwrap_or(false) {
                had_enabled_provider = true;
            }
            if preserve_rounds.is_none() {
                preserve_rounds = meta.rolling_context_preserve_rounds;
            }
            if target.is_none() {
                target = meta.rolling_context_target;
            }
        }
    }

    let final_preserve_rounds = preserve_rounds.unwrap_or(6);
    let final_target = target.unwrap_or(0.6);

    mutate_settings(|settings| {
        // 只要迁移过，就把全局开关显式设置成 true/false，避免后续按默认值歧义解释。
        settings.proxy_rolling_context_enabled = Some(had_enabled_provider);
        settings.proxy_rolling_context_preserve_rounds = Some(final_preserve_rounds);
        settings.proxy_rolling_context_target = Some(final_target);
        settings
            .local_migrations
            .get_or_insert_with(Default::default)
            .proxy_rolling_context_v1 = Some(ProxyRollingContextMigration {
            completed_at: chrono::Utc::now().to_rfc3339(),
            had_enabled_provider,
            preserve_rounds: final_preserve_rounds,
            target: final_target,
        });
    })?;

    Ok(MigrationOutcome {
        skipped_reason: None,
        had_enabled_provider,
        preserve_rounds: final_preserve_rounds,
        target: final_target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use crate::provider::{Provider, ProviderMeta};
    use crate::settings::{get_settings, mutate_settings};
    use serial_test::serial;
    use std::env;
    use tempfile::TempDir;

    struct TempHome {
        #[allow(dead_code)]
        dir: TempDir,
        original_home: Option<String>,
        original_userprofile: Option<String>,
        original_test_home: Option<String>,
    }

    impl TempHome {
        fn new() -> Self {
            let dir = TempDir::new().expect("failed to create temp home");
            let original_home = env::var("HOME").ok();
            let original_userprofile = env::var("USERPROFILE").ok();
            let original_test_home = env::var("CC_SWITCH_TEST_HOME").ok();

            env::set_var("HOME", dir.path());
            env::set_var("USERPROFILE", dir.path());
            env::set_var("CC_SWITCH_TEST_HOME", dir.path());
            crate::settings::reload_settings().expect("reload settings");

            Self {
                dir,
                original_home,
                original_userprofile,
                original_test_home,
            }
        }
    }

    impl Drop for TempHome {
        fn drop(&mut self) {
            match &self.original_home {
                Some(value) => env::set_var("HOME", value),
                None => env::remove_var("HOME"),
            }

            match &self.original_userprofile {
                Some(value) => env::set_var("USERPROFILE", value),
                None => env::remove_var("USERPROFILE"),
            }

            match &self.original_test_home {
                Some(value) => env::set_var("CC_SWITCH_TEST_HOME", value),
                None => env::remove_var("CC_SWITCH_TEST_HOME"),
            }
        }
    }

    fn reset_migration_marker() {
        let _ = mutate_settings(|settings| {
            if let Some(migrations) = settings.local_migrations.as_mut() {
                migrations.proxy_rolling_context_v1 = None;
            }
            settings.proxy_rolling_context_enabled = None;
            settings.proxy_rolling_context_preserve_rounds = None;
            settings.proxy_rolling_context_target = None;
        });
    }

    fn make_provider(enabled: bool, preserve: Option<u32>, target: Option<f64>) -> Provider {
        Provider {
            id: "p1".to_string(),
            name: "test".to_string(),
            settings_config: serde_json::Value::Null,
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            icon: None,
            icon_color: None,
            meta: Some(ProviderMeta {
                context_window: None,
                rolling_context_enabled: Some(enabled),
                rolling_context_threshold: None,
                rolling_context_preserve_rounds: preserve,
                rolling_context_target: target,
                native_auto_compact_enabled: None,
                native_auto_compact_pct: None,
                native_auto_compact_window: None,
                ..Default::default()
            }),
            in_failover_queue: false,
        }
    }

    #[test]
    #[serial]
    fn migration_is_idempotent() {
        let _home = TempHome::new();
        reset_migration_marker();
        let db = Database::memory().unwrap();
        let outcome1 = maybe_migrate_provider_rolling_context_to_global(&db).unwrap();
        assert!(outcome1.skipped_reason.is_none());
        let outcome2 = maybe_migrate_provider_rolling_context_to_global(&db).unwrap();
        assert_eq!(outcome2.skipped_reason.as_deref(), Some("already_migrated"));
    }

    #[test]
    #[serial]
    fn migration_enables_global_when_provider_enabled() {
        let _home = TempHome::new();
        reset_migration_marker();
        let db = Database::memory().unwrap();
        let mut provider = make_provider(true, Some(8), Some(0.5));
        provider.id = "p-enabled".to_string();
        db.save_provider("claude", &provider).unwrap();

        let outcome = maybe_migrate_provider_rolling_context_to_global(&db).unwrap();
        assert!(outcome.had_enabled_provider);
        let settings = get_settings();
        assert_eq!(settings.proxy_rolling_context_enabled, Some(true));
        assert_eq!(settings.proxy_rolling_context_preserve_rounds, Some(8));
        assert_eq!(settings.proxy_rolling_context_target, Some(0.5));
    }
}
