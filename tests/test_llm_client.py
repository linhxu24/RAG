from app.config import Settings
from app.generation.llm_client import OllamaLLMClient, OpenAILLMClient, build_llm_client


def test_llm_client_factory_defaults_to_ollama():
    settings = Settings()

    assert isinstance(build_llm_client(settings), OllamaLLMClient)
    assert settings.llm_router_model == settings.ollama_router_model
    assert settings.llm_generation_model == settings.ollama_generation_model


def test_llm_client_factory_supports_openai_provider():
    settings = Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        openai_router_model="gpt-4.1-nano",
        openai_generation_model="gpt-4.1",
    )

    assert isinstance(build_llm_client(settings), OpenAILLMClient)
    assert settings.llm_router_model == "gpt-4.1-nano"
    assert settings.llm_generation_model == "gpt-4.1"
    assert settings.llm_router_num_ctx is None


def test_provider_role_timeouts_are_positive_even_when_env_uses_zero():
    ollama_settings = Settings(
        llm_provider="ollama",
        ollama_timeout_seconds=0,
        ollama_router_timeout_seconds=0,
        ollama_generation_timeout_seconds=0,
    )
    openai_settings = Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        openai_timeout_seconds=0,
        openai_router_timeout_seconds=0,
        openai_generation_timeout_seconds=0,
    )

    assert ollama_settings.llm_router_timeout_seconds > 0
    assert ollama_settings.llm_generation_timeout_seconds > 0
    assert ollama_settings.ollama_request_timeout_seconds > 0
    assert openai_settings.llm_router_timeout_seconds > 0
    assert openai_settings.llm_generation_timeout_seconds > 0
    assert openai_settings.openai_request_timeout_seconds > 0
