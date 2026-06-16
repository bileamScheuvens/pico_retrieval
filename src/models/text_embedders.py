from src.constants import CACHEPATH
from joblib import Memory
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import lightning as L
import string


class PromptRepsModel(L.LightningModule):
    def __init__(self, model_id):
        super().__init__()
        self.stopwords = set(stopwords.words("english") + list(string.punctuation))

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

        def _get_promptrep(text):
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that can understand human language.",
                },
                {
                    "role": "user",
                    "content": f'Passage: "{text}". Use one word to represent the passage in a retrieval task.',
                },
                {"role": "assistant", "content": 'The word is "'},
            ]

            batch_enc = self.tokenizer.apply_chat_template(  # ty:ignore[unresolved-attribute]
                messages,
                add_generation_prompt=False,
                return_tensors="pt",
            )
            # the last special token is removed
            input_ids = batch_enc["input_ids"][:, :-1].to(self.device)  # ty:ignore[invalid-argument-type, not-subscriptable]

            outputs = self.model(
                input_ids=input_ids, return_dict=True, output_hidden_states=True
            )

            # dense representation
            next_token_reps = outputs.hidden_states[-1][:, -1, :][0]

            # sparse representation
            next_token_logits = torch.log(1 + torch.relu(outputs.logits))[:, -1, :][0]
            return next_token_reps

        self.memory = Memory(CACHEPATH, verbose=0)
        self.get_promptrep = self.memory.cache(_get_promptrep)

    def forward(self, text):
        return self.get_promptrep(text)

    @property
    def embed_dim(self):
        return self.model.config.hidden_size


class SentenceTransformerModel(L.LightningModule):
    def __init__(self, model_id):
        super().__init__()
        self.model = SentenceTransformer(model_id)

    @torch.no_grad
    def forward(self, text):
        return (
            self.model.encode(text, show_progress_bar=False, convert_to_tensor=True)
            .clone()
            .to(self.device)
        )

    @property
    def embed_dim(self):
        return self.model.get_embedding_dimension()
