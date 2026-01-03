from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf

_CFG: Optional[Dict[str, Any]] = None

def init_cfg(cfg: DictConfig | Dict[str, Any]) -> None:
    """
    Initialize global config exactly once.
    Accepts either DictConfig (Hydra) or plain dict (Modal-safe).
    """
    global _CFG
    if _CFG is not None:
        return  # idempotent

    if isinstance(cfg, DictConfig):
        _CFG = OmegaConf.to_container(cfg, resolve=True)
    else:
        _CFG = cfg

def cfg() -> Dict[str, Any]:
    if _CFG is None:
        raise RuntimeError(
            "Global config not initialized. "
            "Call init_cfg(cfg) exactly once (e.g. in main or Modal entry)."
        )
    return _CFG
