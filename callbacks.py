from ast import Dict
from typing import Any
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler


class AgentCallbackHandler(BaseCallbackHandler):
    
  def on_llm_start(
      self,
      serialized: dict[str, Any],
      prompts: list[str],
      *,
      run_id: UUID,
      parent_run_id: UUID | None = None,
      tags: list[str] | None = None,
      metadata: dict[str, Any] | None = None,
      **kwargs: Any
  ) -> Any:
      return super().on_llm_start(
          serialized,
          prompts,
          run_id=run_id,
          parent_run_id=parent_run_id,
          tags=tags,
          metadata=metadata,
          **kwargs
      )

  def on_llm_end(
      self,
      outputs: dict[str, Any],
      *,
      run_id: UUID,
      parent_run_id: UUID | None = None,
      tags: list[str] | None = None,
      metadata: dict[str, Any] | None = None,
      **kwargs: Any
  ) -> Any:
      return super().on_llm_end(outputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)