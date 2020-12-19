from presidio_analyzer import LocalRecognizer, AnalysisExplanation, RecognizerResult


class BirthdayRecognizer(LocalRecognizer):
    ENTITIES = ["BIRTHDAY"]
    MIN_SCORE_WITH_CONTEXT_SIMILARITY = 0.7
    CONTEXT_PREFIX_COUNT = 5
    CONTEXT_SUFFIX_COUNT = 5
    CHECK_LABEL_GROUPS = [
        ({"BIRTHDAY"}, {"DATE"})
    ]

    DEFAULT_EXPLANATION = "Identified as Birthday due to surrounding context words"
    CONTEXT = ["born", "birthday", "birth", "dob"]

    def __init__(self,
                 supported_language="en",
                 supported_entities=None,
                 ner_strength=0.0,
                 check_label_groups=None,
                 context=None):
        supported_entities = supported_entities if supported_entities else self.ENTITIES
        self.ner_strength = ner_strength
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )

        context = context if context else self.CONTEXT
        self.context = context
        super().__init__(supported_entities, supported_language)

    def load(self):
        pass

    @staticmethod
    def build_spacy_explanation(recognizer_name, original_score, explanation):
        explanation = AnalysisExplanation(
            recognizer=recognizer_name,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    def analyze(self, text, entities, nlp_artifacts):
        results = []
        if not nlp_artifacts:
            self.logger.warning("Skipping Birthday, nlp artifacts not provided...")
            return results

        ner_entities = nlp_artifacts.entities
        for entity in entities:
            if entity not in self.supported_entities:
                continue
            for ent in ner_entities:
                if not self.__check_label(entity, ent.label_, self.check_label_groups):
                    continue

                textual_explanation = self.DEFAULT_EXPLANATION
                explanation = self.build_spacy_explanation(
                    self.__class__.__name__, self.ner_strength, textual_explanation
                )
                result = RecognizerResult(
                    entity, ent.start_char, ent.end_char, self.ner_strength, explanation
                )
                results.append(result)

        enhanced_results = \
            self.enhance_using_context(
                text, results, nlp_artifacts, self.context)

        return enhanced_results

    @staticmethod
    def __check_label(entity, label, check_label_groups):
        return any(
            [entity in egrp and label in lgrp for egrp, lgrp in check_label_groups]
        )
