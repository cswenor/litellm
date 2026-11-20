//! The gateway's answers to router-core's hostcalls, so it routes with no Python behind it, the
//! native counterpart of python-bridge's `PythonRouterHost`. [`NativeBootHost`] answers the
//! deployment build at boot, and [`NativeHost`] one inference call: router-core's hostcalls
//! through [`NativeRouterHost`], and the callback hooks through the [`Callbacks`] it holds.

pub mod auth;
mod boot;
mod capacity;
mod model_info;
mod precall;
mod prompt_estimate;
mod router;

pub use boot::NativeBootHost;
pub use capacity::NativeCapacityHost;
pub use router::NativeRouterHost;

use std::sync::Arc;

use litellm_gateway_auth::AuthenticatedRequest;
use litellm_gateway_inference::routing::{Host, RouteAttempt};
use litellm_model_catalog::Catalog;
use litellm_router_core::Deployment;

use crate::{Gateway, callbacks, callbacks::Callbacks};

/// The host the gateway's routes are served through, over the cost map it reads each deployment's
/// context window from.
pub struct NativeHost {
    callbacks: Arc<Callbacks>,
    catalog: Arc<Catalog>,
}

impl NativeHost {
    pub fn new(callbacks: Arc<Callbacks>, catalog: Arc<Catalog>) -> Self {
        Self { callbacks, catalog }
    }

    /// The context window the pre-call checks hold a prompt for `deployment` to.
    pub(crate) async fn window(&self, deployment: &Deployment) -> Option<i64> {
        precall::window(&self.catalog, deployment).await
    }
}

impl Host for NativeHost {
    type RouterHost<'a, A: RouteAttempt + 'a> = NativeRouterHost<'a, A>;
    type Hooks = callbacks::Call;

    fn hooks(&self, identity: &AuthenticatedRequest, call_type: &'static str) -> callbacks::Call {
        self.callbacks.begin(identity, call_type)
    }

    fn router_host<'a, A: RouteAttempt>(
        &'a self,
        gateway: &'a Gateway,
        identity: &'a AuthenticatedRequest,
        route: &'a A,
    ) -> NativeRouterHost<'a, A> {
        NativeRouterHost {
            gateway,
            identity,
            route,
            catalog: &self.catalog,
        }
    }
}
