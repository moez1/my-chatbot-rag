from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.yaml"


class AppConfig(BaseModel):
  name: str
  environment: str
  log_level: str


class OpenAIModels(BaseModel):
  chat: str
  embedding: str


class DeepSeekModels(BaseModel):
  chat: str


class ClaudeModels(BaseModel):
  chat: str


class ModelsConfig(BaseModel):
  openai: OpenAIModels
  deepseek: DeepSeekModels
  claude: ClaudeModels


class ProvidersConfig(BaseModel):
  default_chat: str
  default_embeddings: str
  default_rerank: str
  allow_list: List[str]
  models: ModelsConfig


class RoutingConfig(BaseModel):
  sensitive_chat_provider: str
  non_sensitive_chat_provider: str
  fallback_chain: List[str]


class RagConfig(BaseModel):
  top_k: int
  max_context_chunks: int
  require_citations: bool


class StoresConfig(BaseModel):
  enable_split_stores: bool
  personal_store_name: str
  professional_store_name: str


class DatabaseConfig(BaseModel):
  host: str
  port: int
  name: str
  user: str


class Settings(BaseModel):
  app: AppConfig
  providers: ProvidersConfig
  routing: RoutingConfig
  rag: RagConfig
  stores: StoresConfig
  database: DatabaseConfig


@lru_cache()
def get_settings() -> Settings:
  with CONFIG_PATH.open("r", encoding="utf-8") as f:
    raw = yaml.safe_load(f)
  return Settings(**raw)


