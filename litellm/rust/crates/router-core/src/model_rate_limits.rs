//! `enforce_model_rate_limits` (`ModelRateLimitingCheck`), which the core runs itself when
//! `RouterConfig::enforce_model_rate_limits` is on: an attempt on a deployment past its declared
//! `tpm`, `rpm`, `tpd` or `rpd` is refused, on the counters in the router's cache. A host that
//! runs Python's check leaves the config off, so the rule never runs twice.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jiff::civil::Time;
use serde_json::Value;

use crate::{
    cache::{LimitCounter, LimitKind, RouterCache},
    error::{AttemptError, FailureKind},
    health::current_minute,
    registry::Deployment,
};

/// The `optional_pre_call_checks` entry that turns the check on.
pub const SETTING: &str = "enforce_model_rate_limits";

/// The `litellm_params` that hold a deployment's daily limits, which no other rule reads.
pub const DAILY_PARAMS: [&str; 2] = [TPD, RPD];
const TPD: &str = "tpd";
const RPD: &str = "rpd";

/// `RoutingArgs.ttl` and `RoutingArgs.daily_ttl`.
const MINUTE: Duration = Duration::from_secs(60);
const DAY: Duration = Duration::from_secs(86_400);

/// A deployment's declared limits (`_get_deployment_limits`, `_DailyLimitDeployment`).
struct Limits {
    tpm: Option<f64>,
    rpm: Option<f64>,
    tpd: Option<f64>,
    rpd: Option<f64>,
    /// `daily_reset_time`: how far past midnight UTC the daily window starts over.
    reset: Duration,
}

/// How far past midnight UTC `reset` is, and no way past it when a deployment names no time.
fn past_midnight(reset: Option<Time>) -> Duration {
    reset.map_or(Duration::ZERO, |reset| {
        reset.duration_since(Time::midnight()).unsigned_abs()
    })
}

impl Limits {
    fn of(deployment: &Deployment) -> Self {
        let declared = |key: &str| {
            [
                &deployment.extra,
                &deployment.raw_params,
                &deployment.model_info.raw,
            ]
            .iter()
            .find_map(|section| section.get(key).filter(|value| !value.is_null()))
            .and_then(Value::as_f64)
        };
        let daily = |key: &str| deployment.raw_params.get(key).and_then(Value::as_f64);
        Self {
            tpm: declared("tpm"),
            rpm: declared("rpm"),
            tpd: daily(TPD),
            rpd: daily(RPD),
            reset: past_midnight(deployment.params.daily_reset_time),
        }
    }

    /// The seconds since the epoch the daily window is measured from.
    fn shifted(&self, now: SystemTime) -> f64 {
        now.duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            - self.reset.as_secs_f64()
    }

    /// `daily_window_day`: the UTC day the window started on.
    fn day(&self, now: SystemTime) -> String {
        let days = (self.shifted(now) / 86_400.0).floor() as i64;
        jiff::civil::date(1970, 1, 1)
            .checked_add(jiff::Span::new().days(days))
            .map_or_else(
                |_| String::new(),
                |date| date.strftime("%Y-%m-%d").to_string(),
            )
    }

    /// `seconds_until_daily_reset`.
    fn until_reset(&self, now: SystemTime) -> Duration {
        let into_day = self.shifted(now).rem_euclid(86_400.0);
        Duration::from_secs(((86_400.0 - into_day) as u64).max(1))
    }
}

fn counter(deployment: &Deployment, kind: LimitKind, window: String) -> LimitCounter {
    LimitCounter {
        deployment_id: deployment.id.clone(),
        model: deployment.params.model.clone(),
        kind,
        window,
    }
}

/// A count as Python prints the `int` it is.
fn whole(value: f64) -> String {
    match value.fract() == 0.0 && value.abs() < 1e15 {
        true => format!("{}", value as i64),
        false => value.to_string(),
    }
}

/// The `RateLimitError` the check raises, with its message: never retried, tried again after
/// `retry_after` by its caller, and counted against no per-minute requests, since no request was
/// sent (`_refused` reports it `uncounted`).
fn refused(kind: LimitKind, limit: f64, usage: f64, retry_after: Duration) -> AttemptError {
    let name: &'static str = kind.into();
    AttemptError {
        retry_after: Some(retry_after),
        num_retries: Some(0),
        uncounted_request: true,
        detail: Some(
            format!(
                "Model rate limit exceeded. {} limit={}, current usage={}",
                name.to_uppercase(),
                whole(limit),
                whole(usage),
            )
            .into(),
        ),
        ..AttemptError::new(FailureKind::RateLimit, Some(429))
    }
}

/// `ModelRateLimitingCheck.async_pre_call_check`: the deployment's token windows must have room,
/// then its request windows count this request in one round trip, refunded when that goes past a
/// limit. A cache that fails lets the request through.
pub(crate) async fn admit(
    cache: &impl RouterCache,
    deployment: &Deployment,
    now: SystemTime,
) -> Option<AttemptError> {
    let limits = Limits::of(deployment);
    let minute = current_minute(now);
    let day = limits.day(now);
    let used = |kind, window: &str, limit: f64| {
        let counter = counter(deployment, kind, window.to_owned());
        async move { cache.limit_usage(&counter, limit).await.ok().flatten() }
    };
    if let Some(limit) = limits.tpm
        && let Some(usage) = used(LimitKind::Tpm, &minute, limit).await
        && usage >= limit
    {
        return Some(refused(LimitKind::Tpm, limit, usage, MINUTE));
    }
    if let Some(limit) = limits.tpd
        && let Some(usage) = used(LimitKind::Tpd, &day, limit).await
        && usage >= limit
    {
        let retry_after = limits.until_reset(now);
        return Some(refused(LimitKind::Tpd, limit, usage, retry_after));
    }
    let charged: Vec<(LimitCounter, i64, Duration)> = [
        limits
            .rpm
            .map(|_| (counter(deployment, LimitKind::Rpm, minute), 1, MINUTE)),
        limits
            .rpd
            .map(|_| (counter(deployment, LimitKind::Rpd, day), 1, DAY)),
    ]
    .into_iter()
    .flatten()
    .collect();
    if charged.is_empty() {
        return None;
    }
    let counts = cache
        .add_limit_usages(&charged)
        .await
        .ok()
        .flatten()
        .filter(|counts| counts.len() == charged.len())?;
    let count = |kind| {
        charged
            .iter()
            .zip(&counts)
            .find(|((counter, ..), _)| counter.kind == kind)
            .map(|(_, count)| count.trunc())
    };
    let over = [
        (LimitKind::Rpm, limits.rpm, MINUTE),
        (LimitKind::Rpd, limits.rpd, limits.until_reset(now)),
    ]
    .into_iter()
    .find_map(|(kind, limit, retry_after)| {
        let limit = limit?;
        let usage = count(kind)?;
        (usage > limit).then(|| refused(kind, limit, usage, retry_after))
    })?;
    let refund: Vec<_> = charged
        .into_iter()
        .map(|(counter, _, ttl)| (counter, -1, ttl))
        .collect();
    if let Err(error) = cache.add_limit_usages(&refund).await {
        tracing::debug!(%error, "the model rate limit refund was not recorded");
    }
    Some(over)
}

/// `ModelRateLimitingCheck.async_log_success_event`: a success's tokens count against the
/// deployment's minute and day, whether or not it declares a limit.
pub(crate) async fn count_tokens(
    cache: &impl RouterCache,
    deployment: &Deployment,
    total_tokens: u64,
    now: SystemTime,
) {
    let limits = Limits::of(deployment);
    if total_tokens == 0 || deployment.params.model.is_empty() {
        return;
    }
    let tokens = i64::try_from(total_tokens).unwrap_or(i64::MAX);
    for (kind, window, ttl) in [
        (LimitKind::Tpm, current_minute(now), MINUTE),
        (LimitKind::Tpd, limits.day(now), DAY),
    ] {
        if let Err(error) = cache
            .add_limit_usage(&counter(deployment, kind, window), tokens, ttl)
            .await
        {
            tracing::debug!(%error, "Error in ModelRateLimitingCheck.async_log_success_event");
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// 2024-03-01 00:30:00 UTC.
    fn half_past_midnight() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_709_253_000)
    }

    #[rstest]
    #[case::midnight(None, "2024-03-01", 86_400 - 1800)]
    #[case::before_the_reset(Some(jiff::civil::time(1, 0, 0, 0)), "2024-02-29", 1800)]
    #[case::after_the_reset(Some(jiff::civil::time(0, 15, 0, 0)), "2024-03-01", 86_400 - 900)]
    fn the_daily_window_starts_over_at_the_reset_time(
        #[case] reset: Option<Time>,
        #[case] day: &str,
        #[case] until_reset: u64,
    ) {
        let limits = Limits {
            tpm: None,
            rpm: None,
            tpd: None,
            rpd: None,
            reset: past_midnight(reset),
        };
        assert_eq!(
            (
                limits.day(half_past_midnight()),
                limits.until_reset(half_past_midnight())
            ),
            (day.to_owned(), Duration::from_secs(until_reset))
        );
    }
}
