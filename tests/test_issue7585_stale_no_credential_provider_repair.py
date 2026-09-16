"""Regression coverage for stale no-credential provider repair (#7585).

A session whose stored ``model_provider`` belongs to the catalog-backed class
(e.g. ``openrouter``) but whose catalog group is missing -- typically because
no credential is configured -- must not preserve that lane forever: every
agent construction for it re-routed auxiliary calls through the stale paid
provider. #5731's fail-safe (preserve when ownership evidence is ambiguous or
the stored provider may legitimately own unlisted models) must stay intact.
"""

from types import SimpleNamespace

import pytest

import api.routes as routes


def _catalog(*groups):
    return {"groups": list(groups)}


def _group(provider_id, *models):
    return {"provider_id": provider_id, "models": [{"id": model} for model in models]}


def _session(*, model="deepseek/deepseek-v4.1-flash", provider="openrouter"):
    return SimpleNamespace(model=model, model_provider=provider)


def _repair(session, *, profile_provider="nous", **kwargs):
    return routes._repair_foreign_session_model_provider(
        session,
        requested_model=session.model,
        requested_provider=session.model_provider,
        resolved_model=session.model,
        resolved_provider=session.model_provider,
        explicit_model_pick=False,
        profile_provider=profile_provider,
        **kwargs,
    )


def _patch_catalog(monkeypatch, catalog):
    monkeypatch.setattr(routes, "get_available_models", lambda *, prefer_cache=False: catalog)


@pytest.fixture()
def no_openrouter_credential(monkeypatch):
    monkeypatch.setattr(
        routes, "provider_has_usable_credential",
        lambda pid, **_kw: str(pid).strip().lower() != "openrouter",
        raising=False,
    )


@pytest.fixture()
def no_plugin_providers(monkeypatch):
    monkeypatch.setattr(routes, "is_plugin_model_provider", lambda _pid: False, raising=False)


@pytest.fixture()
def no_provider_credentials(monkeypatch):
    """No provider of any kind has a usable API credential.

    Unlike ``no_openrouter_credential`` (which keeps other lanes live), the
    keyless-local cases under test must have *every* lane credential-dead, so a
    lane is preserved only by the legitimately-keyless / base_url rules and not by
    a surviving credential fallback (PR #7594 review).
    """
    monkeypatch.setattr(
        routes, "provider_has_usable_credential",
        lambda _pid, **_kw: False,
        raising=False,
    )


def test_stale_no_credential_provider_cleared_to_single_owner(
    monkeypatch, no_openrouter_credential, no_plugin_providers
):
    """The #7585 repro: openrouter lane survives while the model is Nous-owned."""
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))

    assert _repair(_session()) == "nous"


def test_ambiguous_ownership_still_preserved(
    monkeypatch, no_openrouter_credential, no_plugin_providers
):
    _patch_catalog(monkeypatch, _catalog(
        _group("nous", "deepseek/deepseek-v4.1-flash"),
        _group("other", "deepseek/deepseek-v4.1-flash"),
    ))

    assert _repair(_session()) == "openrouter"


def test_live_stored_credential_still_preserved(
    monkeypatch, no_plugin_providers
):
    monkeypatch.setattr(routes, "provider_has_usable_credential", lambda _pid, **_kw: True, raising=False)
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))

    assert _repair(_session()) == "openrouter"


def test_incomplete_catalog_evidence_still_preserved(
    monkeypatch, no_openrouter_credential, no_plugin_providers
):
    errored = _group("thirdparty", "some-model")
    errored["models_endpoint_error"] = "upstream 502"
    _patch_catalog(monkeypatch, _catalog(
        _group("nous", "deepseek/deepseek-v4.1-flash"),
        errored,
    ))

    assert _repair(_session()) == "openrouter"


def test_self_hosted_stored_provider_keeps_5731_fail_safe(monkeypatch, no_plugin_providers):
    """ollama may legitimately own unlisted models: missing group stays preserved."""
    monkeypatch.setattr(routes, "provider_has_usable_credential", lambda _pid, **_kw: False, raising=False)
    _patch_catalog(monkeypatch, _catalog(_group("kilocode", "kilo/minimax/minimax-m3")))
    session = _session(model="kilo/minimax/minimax-m3", provider="ollama")

    assert _repair(session, profile_provider="kilocode") == "ollama"


def test_plugin_stored_provider_stays_preserved(monkeypatch, no_openrouter_credential):
    monkeypatch.setattr(routes, "is_plugin_model_provider", lambda _pid: True, raising=False)
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))

    assert _repair(_session()) == "openrouter"


def test_chat_start_no_longer_routes_agent_through_stale_paid_lane(
    monkeypatch, tmp_path, no_openrouter_credential, no_plugin_providers
):
    """End-to-end billing guard: the stale openrouter lane must not reach a run."""
    session = SimpleNamespace(
        session_id="issue-7585",
        workspace=str(tmp_path),
        model="deepseek/deepseek-v4.1-flash",
        model_provider="openrouter",
        profile="default",
        messages=[],
        context_messages=[],
        pending_user_message=None,
        save=lambda: None,
    )
    captured = {}

    def start_run(s, **kwargs):
        captured.update(kwargs)
        routes._prepare_chat_start_session_for_stream(
            s,
            msg=kwargs["msg"],
            attachments=kwargs["attachments"],
            workspace=kwargs["workspace"],
            model=kwargs["model"],
            model_provider=kwargs["model_provider"],
            stream_id="issue-7585-stream",
        )
        return {"stream_id": "issue-7585-stream"}

    monkeypatch.setattr(routes, "_get_or_materialize_session", lambda _sid, **_kwargs: session)
    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {"model": {"provider": "nous"}}))
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))
    monkeypatch.setattr(routes, "_start_run", start_run)
    monkeypatch.setattr(routes, "j", lambda _handler, payload, status=200: payload)

    routes._handle_chat_start(None, {"session_id": session.session_id, "message": "continue"})

    assert captured["model_provider"] == "nous", captured["model_provider"]
    assert session.model == "deepseek/deepseek-v4.1-flash"
    assert session.model_provider == "nous"


def test_minimal_fallback_catalog_stays_preserved(
    monkeypatch, no_openrouter_credential, no_plugin_providers
):
    """A cold/emergency minimal catalog is not complete discovery evidence.

    The fallback lists only the active provider's default model; treating it
    as a full catalog could reassign the session to the wrong owner when
    other configured providers were merely omitted (PR #7594 review).
    """
    minimal = _catalog(_group("nous", "deepseek/deepseek-v4.1-flash"))
    minimal["catalog_minimal"] = True
    _patch_catalog(monkeypatch, minimal)

    assert _repair(_session()) == "openrouter"


def test_keyless_custom_endpoint_provider_stays_preserved(
    monkeypatch, no_openrouter_credential, no_plugin_providers
):
    """custom:<slug> lanes (vLLM, llama-server) need no API key.

    A missing catalog group proves nothing for them: absence of a key must
    not be read as proof of staleness (PR #7594 review).
    """
    monkeypatch.setattr(routes, "provider_has_usable_credential", lambda _pid, **_kw: False, raising=False)
    _patch_catalog(monkeypatch, _catalog(_group("nous", "vendor/local-model")))
    session = _session(model="vendor/local-model", provider="custom:vllm-local")

    assert _repair(session, profile_provider="nous") == "custom:vllm-local"


# ---------------------------------------------------------------------------
# PR #7594 review (CHANGES_REQUESTED): keyless LOCAL-server lanes beyond
# ollama/lmstudio and any configured ``providers.<id>.base_url`` OpenAI-compatible
# endpoint must be preserved even with no catalog group and no API key. The #7585
# repair must never silently reassign a working self-hosted session to the catalog
# owner, nor persist such a swap.
# ---------------------------------------------------------------------------


def test_raw_keyless_vllm_provider_preserved_without_catalog(
    monkeypatch, no_provider_credentials, no_plugin_providers
):
    """A raw routable ``vllm`` lane (no catalog group, no key) must stay put.

    vllm is a local model server (`_is_local_server_provider`): its models never
    appear in any catalog, so the *absence* of a vllm group proves nothing.
    Clearing it here would silently reassign a working self-hosted session to the
    catalog owner (PR #7594 review, CORE).
    """
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))

    assert _repair(_session(provider="vllm")) == "vllm"


def test_raw_keyless_llamacpp_provider_preserved_without_catalog(
    monkeypatch, no_provider_credentials, no_plugin_providers
):
    """A raw ``llamacpp`` lane is likewise self-hosted and must be preserved."""
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))

    assert _repair(_session(provider="llamacpp")) == "llamacpp"


def test_raw_keyless_tabby_provider_preserved_without_catalog(
    monkeypatch, no_provider_credentials, no_plugin_providers
):
    """``tabby`` (TabbyAPI) is another local server name in _LOCAL_SERVER_PROVIDERS."""
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))

    assert _repair(_session(provider="tabby")) == "tabby"


def _run_chat_start_local_lane(monkeypatch, tmp_path, *, session_id, provider, profile_cfg):
    """Drive a real chat start for a session whose stored provider is ``provider``.

    Returns (captured_kwargs_passed_to__start_run, session_provider_after, provider_before).
    ``profile_cfg`` is the per-profile config returned by the (patched)
    ``_read_profile_model_config``; it must keep the ``nous`` catalog owning the
    stored model so a replacement would target it.
    """
    session = SimpleNamespace(
        session_id=session_id,
        workspace=str(tmp_path),
        model="deepseek/deepseek-v4.1-flash",
        model_provider=provider,
        profile="default",
        messages=[],
        context_messages=[],
        pending_user_message=None,
        save=lambda: None,
    )
    provider_before = session.model_provider
    captured = {}

    def start_run(s, **kwargs):
        captured.update(kwargs)
        routes._prepare_chat_start_session_for_stream(
            s,
            msg=kwargs["msg"],
            attachments=kwargs["attachments"],
            workspace=kwargs["workspace"],
            model=kwargs["model"],
            model_provider=kwargs["model_provider"],
            stream_id=session_id,
        )
        return {"stream_id": session_id}

    monkeypatch.setattr(routes, "_get_or_materialize_session", lambda _sid, **_kwargs: session)
    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(
        routes,
        "_read_profile_model_config",
        lambda _s, _p: (None, None, profile_cfg or {"model": {"provider": "nous"}}),
    )
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))
    monkeypatch.setattr(routes, "_start_run", start_run)
    monkeypatch.setattr(routes, "j", lambda _handler, payload, status=200: payload)

    routes._handle_chat_start(None, {"session_id": session.session_id, "message": "continue"})

    return captured, session.model_provider, provider_before


def test_chat_start_keeps_keyless_local_vllm_lane_and_does_not_persist(
    monkeypatch, tmp_path, no_provider_credentials, no_plugin_providers
):
    """End-to-end: a keyless raw ``vllm`` lane passes through chat start untouched.

    The stored provider must not be silently swapped for the catalog owner and must
    not be persisted as a different provider. Provider is identical before/after.
    """
    captured, provider_after, provider_before = _run_chat_start_local_lane(
        monkeypatch, tmp_path, session_id="issue-7594-vllm", provider="vllm",
        profile_cfg=None,
    )

    assert captured["model_provider"] == "vllm", captured["model_provider"]
    assert provider_after == "vllm", provider_after
    assert provider_before == provider_after


def test_chat_start_keeps_configured_openai_compatible_base_url_lane(
    monkeypatch, tmp_path, no_provider_credentials, no_plugin_providers
):
    """Any profile-scoped ``providers.<id>.base_url`` OpenAI-compatible endpoint is
    preserved even with no catalog group and no key.

    ``llama-server`` here is an arbitrary OpenAI-compatible id declared via
    ``providers.llama-server.base_url``; its models are served without any catalog.
    Chat start must keep the lane identical (no replace, no persist).
    """
    profile_cfg = {
        "providers": {
            "llama-server": {"base_url": "http://127.0.0.1:8080/v1"},
        },
        "model": {"provider": "nous"},
    }
    captured, provider_after, provider_before = _run_chat_start_local_lane(
        monkeypatch, tmp_path, session_id="issue-7594-burl", provider="llama-server",
        profile_cfg=profile_cfg,
    )

    assert captured["model_provider"] == "llama-server", captured["model_provider"]
    assert provider_after == "llama-server", provider_after
    assert provider_before == provider_after


# ---------------------------------------------------------------------------
# PR #7594 review P1 (follow-up): TOP-LEVEL ``model.base_url`` local lane.
# A profile config may declare the local endpoint at the TOP level
# (FAQ-documented shape ``model: {provider: ..., base_url: http://127.0.0.1:...}``)
# rather than nested under ``providers.<id>.base_url`` (which the previous fix
# already honors). An arbitrary provider ID -- e.g. ``my-local-server`` -- routed
# through a top-level loopback / private base_url, with no key and no catalog
# group, must be preserved by the stale-provider repair: chat start must not
# reassign it to the catalog owner and must not persist such a swap. The nested
# providers.<id>.base_url check alone (already in production) does not catch the
# top-level shape, so these tests pin the missing P1 lane.
# ---------------------------------------------------------------------------


def _repair_with_top_level(session, profile_cfg, *, provider="nous"):
    return routes._repair_foreign_session_model_provider(
        session,
        requested_model=session.model,
        requested_provider=session.model_provider,
        resolved_model=session.model,
        resolved_provider=session.model_provider,
        explicit_model_pick=False,
        profile_provider=provider,
        profile_config=profile_cfg,
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
        "http://192.168.1.50:1234/v1",
        "http://10.0.0.7:8080/v1",
    ],
)
def test_top_level_model_base_url_arbitrary_provider_preserved_without_catalog(
    monkeypatch, no_provider_credentials, no_plugin_providers, base_url
):
    """A TOP-LEVEL model.base_url (loopback/private) must preserve an arbitrary
    stored provider ID even with no catalog group and no API key (P1)."""
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))
    profile_config = {"model": {"provider": "nous", "base_url": base_url}}
    session = _session(provider="my-local-server")

    result = _repair_with_top_level(session, profile_config)

    assert result == "my-local-server", result


def test_top_level_model_base_url_arbitrary_provider_preserved_when_default_provider_matches(
    monkeypatch, no_provider_credentials, no_plugin_providers
):
    """Same lane preserved when the profile default provider IS the arbitrary id.

    The top-level model.provider being the stored provider is the FAQ-documented
    setup; a top-level base_url still must not let the repair clear it.
    """
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))
    profile = {"model": {"provider": "my-local-server", "base_url": "http://127.0.0.1:8000/v1"}}
    session = _session(provider="my-local-server")

    result = _repair_with_top_level(session, profile, provider="my-local-server")

    assert result == "my-local-server", result


def test_chat_start_keeps_top_level_base_url_lane_and_does_not_persist(
    monkeypatch, tmp_path, no_provider_credentials, no_plugin_providers
):
    """End-to-end: a session stored under an arbitrary provider ID whose profile
    declares a TOP-LEVEL model.base_url loopback passes through chat start
    untouched -- no swap to the catalog owner, no persisted rewrite."""
    profile_cfg = {"model": {"provider": "nous", "base_url": "http://127.0.0.1:8080/v1"}}
    captured, provider_after, provider_before = _run_chat_start_local_lane(
        monkeypatch, tmp_path, session_id="issue-7594-topburl", provider="my-local-server",
        profile_cfg=profile_cfg,
    )

    assert captured["model_provider"] == "my-local-server", captured["model_provider"]
    assert provider_after == "my-local-server", provider_after
    assert provider_before == provider_after


def test_top_level_public_base_url_does_not_preserve_stale_lane(
    monkeypatch, no_provider_credentials, no_plugin_providers
):
    """A NON-loopback/private top-level base_url is not a local-routing signal.

    The P1 contract pins loopback/private base_url URLs (127.0.0.1, localhost,
    RFC1918 private, 10.x) as local-host evidence. A public endpoint (here a
    vendor-relay-ish URL) with no key and no catalog group must still be
    treated as a stale catalog-backed lane and reassigned to the single catalog
    owner -- not blanket-preserved by any base_url value.
    """
    _patch_catalog(monkeypatch, _catalog(_group("nous", "deepseek/deepseek-v4.1-flash")))
    profile_config = {
        "model": {"provider": "nous", "base_url": "https://relay.example.com/v1"}
    }
    session = _session(provider="my-local-server")

    result = _repair_with_top_level(session, profile_config)

    assert result == "nous", result
