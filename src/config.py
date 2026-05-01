from pathlib import Path
from pydantic_settings import SettingsConfigDict, BaseSettings, PydanticBaseSettingsSource, TomlConfigSettingsSource, DotEnvSettingsSource


class Settings(BaseSettings):

    db_dir: str
    collection_name: str
    qdrant_vector_size: int
    qdrant_url: str

    openai_api_key: str
    llm_model: str
    llm_base_url: str
    embedding_model: str

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_chunks: int = 5
    
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