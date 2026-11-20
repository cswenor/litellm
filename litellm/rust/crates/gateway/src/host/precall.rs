//! The facts `enable_pre_call_checks` decides by (`litellm/router_rust/steps/precall.py`): each
//! candidate's context window as `get_router_model_info` merges it, its own
//! `model_info.max_input_tokens` over the cost map's, and the tokens of the prompt the route
//! hands over, as [`prompt_estimate`](super::prompt_estimate) estimates them. router-core decides
//! with them, and a route that hands over no prompt is held to no window.
//!
//! No region and no requested param is reported. A region comes from the caller's key and the one
//! param the proxy filters on is `response_format`, and `admit` leaves a key that carries the
//! first and a Messages body that names the second to Python.

use std::sync::Arc;

use futures_util::future;
use litellm_gateway_inference::routing::Prompt;
use litellm_model_catalog::Catalog;
use litellm_router_core::{
    Deployment,
    model_info::{RouterModelInfoQuery, router_model_info},
    precall::{DeploymentFacts, PreCallFacts},
};
use serde_json::Value;

use super::{model_info::NativeModelInfoHost, prompt_estimate::Countable};

/// `_deployment_max_input_tokens`: the merged context window, or `None` when there is none or the
/// deployment cannot be resolved. It is read from the deployment alone, so `admit` reads the same
/// window at boot that a call is held to.
pub(super) async fn window(catalog: &Catalog, deployment: &Deployment) -> Option<i64> {
    let base_model = |named: Option<&Value>| named.cloned().unwrap_or(Value::Null);
    let query = RouterModelInfoQuery {
        deployment_id: Value::String(deployment.id.clone()),
        model_info_base_model: base_model(deployment.model_info.raw.get("base_model")),
        params_base_model: base_model(deployment.raw_params.get("base_model")),
        params_model: deployment.params.model.clone(),
        extra_base_model: base_model(deployment.params.extra.get("base_model")),
        received_model_name: deployment.model_name.clone(),
    };
    let host = NativeModelInfoHost {
        catalog,
        deployment,
    };
    match router_model_info(&host, &query).await {
        Ok(merged) => merged?.info.get("max_input_tokens")?.as_i64(),
        Err(error) => {
            tracing::debug!(%error, id = %deployment.id, "skipping the deployment's context window");
            None
        }
    }
}

/// The estimate runs off the event loop, since it tokenizes the whole prompt.
async fn tokens(prompt: &Prompt<'_>) -> Option<i64> {
    let countable = Countable::of(prompt)?;
    tokio::task::spawn_blocking(move || countable.tokens())
        .await
        .ok()
        .flatten()
}

pub(super) async fn facts(
    catalog: &Catalog,
    prompt: Option<Prompt<'_>>,
    candidates: &[Arc<Deployment>],
) -> PreCallFacts {
    let Some(prompt) = prompt else {
        return PreCallFacts::default();
    };
    let windows = future::join_all(candidates.iter().map(async |deployment| {
        let facts = DeploymentFacts {
            max_input_tokens: window(catalog, deployment).await,
            ..DeploymentFacts::default()
        };
        (deployment.id.clone(), facts)
    }))
    .await;
    let has_window = windows
        .iter()
        .any(|(_, facts)| facts.max_input_tokens.is_some());
    PreCallFacts {
        has_input: true,
        countable: true,
        input_tokens: if has_window {
            tokens(&prompt).await
        } else {
            None
        },
        allowed_model_region: None,
        requested_params: Vec::new(),
        deployments: windows.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use litellm_router_core::DeploymentSpec;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const COST_MAP: &[u8] = br#"{"listed-model": {"litellm_provider": "anthropic", "mode": "chat", "max_input_tokens": 5000}}"#;

    #[rstest]
    #[case::its_own_over_the_cost_maps("anthropic/listed-model", json!({"max_input_tokens": 100}), Some(100))]
    #[case::the_cost_maps("anthropic/listed-model", json!({}), Some(5000))]
    #[case::its_own_for_a_model_the_cost_map_does_not_list(
        "anthropic/in-house-9",
        json!({"max_input_tokens": 100}),
        Some(100)
    )]
    #[case::none_for_a_model_the_cost_map_does_not_list("anthropic/in-house-9", json!({}), None)]
    #[tokio::test]
    async fn a_deployments_window_is(
        #[case] model: &str,
        #[case] model_info: Value,
        #[case] expected: Option<i64>,
    ) {
        let catalog = Catalog::parse(COST_MAP).unwrap().value;
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "claude",
            "litellm_params": {"model": model},
            "model_info": model_info,
        }))
        .unwrap();
        let deployment = Deployment::from_spec(spec).unwrap();

        assert_eq!(window(&catalog, &deployment).await, expected);
    }
}
