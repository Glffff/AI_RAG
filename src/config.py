from pathlib import Path
from pydantic_settings import SettingsConfigDict, BaseSettings, PydanticBaseSettingsSource, TomlConfigSettingsSource, DotEnvSettingsSource


class Settings(BaseSettings):
    
    db_dir: str
    openai_api_key: str
    llm_provider: str = "local"
    embeddings_provider: str = "local"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_chunks: int = 5
    ollama_model: str = "mistral"
    ollama_base_url: str = "http://localhost:11434"
    local_embeddings_model: str = "all-MiniLM-L6-v2"
    llm_temperature: float = 0.7
    
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        extra="ignore"
    )
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        toml_file = Path(__file__).parent.parent / "configs" / "local-settings.toml"
        
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=toml_file),
        )

settings = Settings() #type: ignore