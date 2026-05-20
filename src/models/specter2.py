from transformers import AutoTokenizer
from adapters import AutoAdapterModel


class SPECTER2Model:
    def __init__(self):
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

        # load base model
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

        self.model.load_adapter(
            "allenai/specter2_regression",
            source="hf",
            load_as="proximity",
            set_active=True,
        )

    def get_embedding(self, papers):
        text_batch = [
            d["title"] + self.tokenizer.sep_token + (d.get("abstract") or "")
            for d in papers
        ]
        inputs = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        output = self.model(**inputs)
        # take the first token in the batch as the embedding
        return output.last_hidden_state[:, 0, :]
