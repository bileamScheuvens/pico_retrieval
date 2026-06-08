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

    def forward(self, text):
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

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt"
        )[:, :-1].to(self.device)  # the last special token is removed

        outputs = self.model(
            input_ids=input_ids, return_dict=True, output_hidden_states=True
        )

        # dense representation
        next_token_reps = outputs.hidden_states[-1][:, -1, :][0]

        # sparse representation
        next_token_logits = torch.log(1 + torch.relu(outputs.logits))[:, -1, :][0]
        return next_token_reps

    @property
    def embed_dim(self):
        # TODO
        raise NotImplementedError


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
        return self.model.get_sentence_embedding_dimension()
