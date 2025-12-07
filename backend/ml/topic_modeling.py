# backend/ml/topic_modeling.py
"""
Simple topic modeling utilities using Gensim LDA.
"""

from typing import Iterable, List, Optional, Tuple

from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.preprocessing import STOPWORDS

DEFAULT_TOPICS = 5


def preprocess_docs(docs: Iterable[str]) -> List[List[str]]:
    """Lowercase, whitespace tokenization, drop stopwords."""
    processed: List[List[str]] = []
    for doc in docs:
        if not doc:
            processed.append([])
            continue
        tokens = [
            token
            for token in doc.lower().split()
            if token not in STOPWORDS and token.isalpha()
        ]
        processed.append(tokens)
    return processed


def prepare_corpus(docs: Iterable[str]):
    """
    Preprocess docs, build bigrams, and dictionary/corpus once for reuse.
    """
    tokenized = preprocess_docs(docs)
    if not any(tokenized):
        return [], None, []

    phrases = Phrases(tokenized, min_count=2, threshold=4)
    bigram = Phraser(phrases)
    tokenized = [bigram[doc] for doc in tokenized]

    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=1, no_above=0.6, keep_n=5000)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    return tokenized, dictionary, corpus


def train_lda(
    corpus,
    dictionary: corpora.Dictionary,
    num_topics: int,
    passes: int = 8,
) -> Optional[models.LdaModel]:
    if dictionary is None or len(dictionary) == 0 or all(len(vec) == 0 for vec in corpus):
        return None

    adjusted_topics = max(1, min(num_topics, len(dictionary)))
    return models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=adjusted_topics,
        random_state=42,
        passes=passes,
    )


def build_lda_model(
    docs: Iterable[str],
    num_topics: int = DEFAULT_TOPICS,
    passes: int = 8,
) -> Tuple[Optional[models.LdaModel], Optional[corpora.Dictionary], List[List[tuple]]]:
    """
    Train a small LDA model and return model, dictionary, and corpus.
    """
    tokenized, dictionary, corpus = prepare_corpus(docs)
    if dictionary is None:
        return None, None, []

    lda_model = train_lda(corpus, dictionary, num_topics=num_topics, passes=passes)
    return lda_model, dictionary, corpus


def describe_topics(
    lda_model: models.LdaModel, num_words: int = 5
) -> List[List[Tuple[str, float]]]:
    """Return top words per topic."""
    return lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)


def infer_topics(
    lda_model: models.LdaModel,
    dictionary: corpora.Dictionary,
    docs: Iterable[str],
) -> List[List[Tuple[int, float]]]:
    """
    Get per-document topic distributions.
    """
    tokenized = preprocess_docs(docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    return [lda_model.get_document_topics(vec) for vec in corpus]
