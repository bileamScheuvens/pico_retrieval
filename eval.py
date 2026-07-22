import hydra
from omegaconf import DictConfig

from src.constants import CONFIGPATH
from src.eval.eval_dash import eval_dash
from src.eval.eval_ebm_nlp import eval_ebm_nlp
from src.eval.eval_probe import eval_probe
from src.eval.eval_visualisation import eval_visualisation
from src.eval.eval_transfer import eval_transfer
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
    raise NotImplementedError


if __name__ == "__main__":
    eval()
