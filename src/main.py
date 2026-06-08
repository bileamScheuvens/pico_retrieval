from omegaconf import DictConfig
from src.constants import CONFIGPATH
import hydra
from src.train import train_artsy

#
# papers = [
#     {
#         "title": "Steroids in pregnant women",
#         "abstract": "Long-term outcomes after repeat doses of antenatal corticosteroids. Previous trials have shown that repeat courses of antenatal corticosteroids improve some neonatal outcomes in preterm infants but reduce birth weight and increase the risk of intrauterine growth restriction. We report long-term follow-up results of children enrolled in a randomized trial comparing single and repeat courses of antenatal corticosteroids. Women at 23 through 31 weeks of gestation who remained pregnant 7 days after an initial course of corticosteroids were randomly assigned to weekly courses of betamethasone, consisting of 12 mg given intramuscularly and repeated once at 24 hours, or an identical-appearing placebo. ",
#     },
# ]
#


@hydra.main(version_base=None, config_path=str(CONFIGPATH), config_name="config")
def main(cfg: DictConfig):
    train_artsy(cfg)  # ty:ignore[invalid-argument-type]


if __name__ == "__main__":
    main()
