"""
Generic 'typed decision' classifier -- the same idea as Jev's Choice/Score:
give it a handful of labeled examples, it learns a tiny classifier on top of
sentence embeddings, and returns a label + confidence WITHOUT generating any
text and WITHOUT calling the big synthesis LLM. Runs on CPU, trains in
seconds to minutes, fully offline (no API calls, safe for classified data).

Backed by SetFit (few-shot classification on sentence-transformer embeddings).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "trained_models"


@dataclass
class Decision:
    label: str
    confidence: float
    all_scores: dict[str, float]


class FastClassifier:
    """One typed-decision classifier, e.g. 'what type of document is this'
    or 'how urgent is this'. Train once, reuse forever -- cheap enough to
    run on every chunk during ingestion instead of the synthesis LLM."""

    def __init__(self, name: str, base_model: str = "sentence-transformers/paraphrase-mpnet-base-v2"):
        self.name = name
        self.base_model = base_model
        self._model = None  # lazy-loaded

    # -- training --------------------------------------------------------
    def train(self, texts: list[str], labels: list[str], num_epochs: int = 4) -> None:
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset

        model = SetFitModel.from_pretrained(self.base_model)
        dataset = Dataset.from_dict({"text": texts, "label": labels})
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=TrainingArguments(num_epochs=num_epochs, batch_size=16),
        )
        trainer.train()
        self._model = model
        self.save()

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(MODEL_DIR / self.name))

    def load(self) -> None:
        from setfit import SetFitModel

        self._model = SetFitModel.from_pretrained(str(MODEL_DIR / self.name))

    # -- inference ---------------------------------------------------------
    def predict(self, text: str) -> Decision:
        if self._model is None:
            self.load()
        probs = self._model.predict_proba([text])[0]
        labels = self._model.labels or [str(i) for i in range(len(probs))]
        scores = {label: float(p) for label, p in zip(labels, probs)}
        best_label = max(scores, key=scores.get)
        return Decision(label=best_label, confidence=scores[best_label], all_scores=scores)

    def predict_batch(self, texts: list[str]) -> list[Decision]:
        if self._model is None:
            self.load()
        probs = self._model.predict_proba(texts)
        labels = self._model.labels or [str(i) for i in range(len(probs[0]))]
        out = []
        for row in probs:
            scores = {label: float(p) for label, p in zip(labels, row)}
            best_label = max(scores, key=scores.get)
            out.append(Decision(label=best_label, confidence=scores[best_label], all_scores=scores))
        return out
