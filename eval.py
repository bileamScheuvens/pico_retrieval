import hydra
from omegaconf import DictConfig
from src.eval import (
    eval_dash,
    eval_ebm_nlp,
    eval_probe,
    eval_sysrev,
    eval_transfer,
    eval_visualisation,
)
from src.constants import CONFIGPATH
from src.utils.configs import EvalMethods


@hydra.main(version_base=None, config_path=str(CONFIGPATH), config_name="eval")
def eval(cfg: DictConfig):
    if cfg.eval_method == EvalMethods.EBM_NLP:
        return eval_ebm_nlp(cfg)
    if cfg.eval_method == EvalMethods.VIS:
        return eval_visualisation(cfg)
    if cfg.eval_method == EvalMethods.PROBE:
        return eval_probe(cfg)
    if cfg.eval_method == EvalMethods.DASH:
        return eval_dash(cfg)
    if cfg.eval_method == EvalMethods.TRANSFER:
        return eval_transfer(cfg)
    if cfg.eval_method == EvalMethods.SYSREV:
        return eval_sysrev(cfg)
    raise NotImplementedError


if __name__ == "__main__":
    eval()
