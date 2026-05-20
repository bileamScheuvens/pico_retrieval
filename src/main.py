from src.models.pubmed_pico import PubMedPicoModel
from src.models.specter2 import SPECTER2Model

papers = [
    {
        "title": "Steroids in pregnant women",
        "abstract": "Long-term outcomes after repeat doses of antenatal corticosteroids. Previous trials have shown that repeat courses of antenatal corticosteroids improve some neonatal outcomes in preterm infants but reduce birth weight and increase the risk of intrauterine growth restriction. We report long-term follow-up results of children enrolled in a randomized trial comparing single and repeat courses of antenatal corticosteroids. Women at 23 through 31 weeks of gestation who remained pregnant 7 days after an initial course of corticosteroids were randomly assigned to weekly courses of betamethasone, consisting of 12 mg given intramuscularly and repeated once at 24 hours, or an identical-appearing placebo. ",
    },
]


pubmed_pico = PubMedPicoModel()
specter2 = SPECTER2Model()

pico = pubmed_pico.predict_pico(papers[0]["abstract"])
embed = specter2.get_embedding(papers)[0]

print(embed)
print(pico)
