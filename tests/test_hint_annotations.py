import unittest
from tau2_grpo.hint_annotations import validate_annotation

class AnnotationTests(unittest.TestCase):
    def annotation(self,hint):return {'diagnosis':'The required text is missing.','confidence':'high','hint':hint}
    def test_separate_valid_fields(self):
        a=self.annotation('Ask the customer for the missing text before calling the tool.')
        self.assertEqual(validate_annotation(a,False),a)
        self.assertEqual(validate_annotation(self.annotation('Copy the text from the latest reply exactly.'),True)['confidence'],'high')
    def test_wrong_stage_rejected(self):
        with self.assertRaises(ValueError):validate_annotation(self.annotation('Ask for the text again.'),True)
    def test_invalid_confidence_or_missing_diagnosis_rejected(self):
        a=self.annotation('Ask for the missing text.');a['confidence']='certain'
        with self.assertRaises(ValueError):validate_annotation(a,False)
        with self.assertRaises(ValueError):validate_annotation({'hint':'Ask for the text.'},False)
