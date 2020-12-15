import spacy
from sutime import SUTime

from presidio_analyzer import PresidioLogger
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine

logger = PresidioLogger()
spacy.prefer_gpu()


class SpacyNlpEngine(NlpEngine):
    """ SpacyNlpEngine is an abstraction layer over the nlp module.
        It provides processing functionality as well as other queries
        on tokens.
        The SpacyNlpEngine uses SpaCy as its NLP module
    """

    engine_name = "spacy"
    is_available = bool(spacy)

    def __init__(self, models=None, overwrite_date=True):
        if not models:
            models = {"en": "en_core_web_trf"}
        logger.debug(f"Loading SpaCy models: {models.values()}")

        self.nlp = {
            lang_code: spacy.load(model_name, disable=['parser', 'tagger'])
            for lang_code, model_name in models.items()
        }

        self.su_time = SUTime()
        self.overwrite_date = overwrite_date
        for model_name in models.values():
            logger.debug("Printing spaCy model and package details:"
                         "\n\n {}\n\n".format(spacy.info(model_name)))

    def process_text(self, text, language):
        """ Execute the SpaCy NLP pipeline on the given text
            and language
        """
        doc = self.nlp[language](text)
        return self.doc_to_nlp_artifact(doc, language)

    def is_stopword(self, word, language):
        """ returns true if the given word is a stop word
            (within the given language)
        """
        return self.nlp[language].vocab[word].is_stop

    def is_punct(self, word, language):
        """ returns true if the given word is a punctuation word
            (within the given language)
        """
        return self.nlp[language].vocab[word].is_punct

    def get_nlp(self, language):
        return self.nlp[language]

    def doc_to_nlp_artifact(self, doc, language):
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        tokens_indices = [token.idx for token in doc]

        self.parse_date(doc)

        entities = doc.ents

        return NlpArtifacts(entities=entities, tokens=tokens,
                            tokens_indices=tokens_indices, lemmas=lemmas,
                            nlp_engine=self, language=language)

    def parse_date(self, doc):
        su_time = self.su_time
        matches = su_time.parse(doc.text)
        seen_tokens = set()
        entities = list(doc.ents)
        new_entities = []
        for match in matches:
            span = doc.char_span(match["start"], match["end"], label=match["type"])

            if span is None:
                continue

            # There are instances like 1111 is recognized as Date. This ugly hack is to prevent it :(.
            if span.text.isnumeric():
                if len(span.text) > 4:
                    continue
                elif len(span.text) == 4 and int(span.text) < 1900:
                    continue

            start = span.start
            end = span.end

            if any(t.ent_type for t in doc[start:end]) and not self.overwrite_date:
                continue

            if start not in seen_tokens and end - 1 not in seen_tokens:
                new_entities.append(span)
                entities = [
                    e for e in entities if not (e.start < end and e.end > start)
                ]
                seen_tokens.update(range(start, end))

        doc.ents = entities + new_entities
        return doc
