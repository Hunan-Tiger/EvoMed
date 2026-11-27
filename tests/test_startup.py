import argparse
import contextlib
import io
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate
import train
from retrieval.Reranker import MedicalVisualAugmentation
from utils.utils import inference_chat_api_compatible


class StartupTests(unittest.TestCase):
    def test_api_key_redacted_in_both_log_sinks_but_retained_for_inference(self):
        fake_key = "synthetic-startup-test-key"
        for module in (train, evaluate):
            for key_source in (fake_key, "env:EVOMED_TEST_API_KEY"):
                with self.subTest(script=module.__name__, env_key=key_source != fake_key):
                    with tempfile.TemporaryDirectory() as directory:
                        previous_directory = Path.cwd()
                        os.chdir(directory)
                        root = logging.getLogger()
                        previous_handlers, previous_level = root.handlers[:], root.level
                        root.handlers.clear()
                        try:
                            args = argparse.Namespace(
                                api_key=key_source,
                                base_url="https://example.invalid",
                                model_name_or_path="startup-test-model",
                                classifier_model="startup-test-classifier",
                                evaluator_model="startup-test-evaluator",
                                infer_mode="api",
                                exp_name="startup-test",
                                give_knowledges="false",
                                rollout_max_tokens=1,
                                rollout_temperature=0,
                                rollout_top_p=1,
                            )
                            output = io.StringIO()
                            with (
                                contextlib.redirect_stdout(output),
                                patch.dict(os.environ, {"EVOMED_TEST_API_KEY": fake_key}),
                                patch.object(MedicalVisualAugmentation, "__init__", return_value=None) as reranker,
                            ):
                                agent = module.LLM(args)
                            reranker.assert_called_once_with(device="cuda")
                            self.assertIsInstance(agent.reranker, MedicalVisualAugmentation)
                            self.assertEqual(agent.args.api_key, fake_key)
                            with patch("utils.utils.requests.post") as post:
                                post.return_value.json.return_value = {
                                    "choices": [{"message": {"content": "ok"}}],
                                }
                                reply, _ = inference_chat_api_compatible([], agent.args)
                            self.assertEqual(reply, "ok")
                            self.assertEqual(
                                post.call_args.kwargs["headers"]["Authorization"],
                                f"Bearer {fake_key}",
                            )
                            for log in (output.getvalue(), Path(agent.log_file).read_text(encoding="utf-8")):
                                self.assertIn("Arguments:", log)
                                self.assertIn("startup-test-model", log)
                                self.assertIn("[REDACTED]", log)
                                self.assertNotIn(fake_key, log)
                        finally:
                            for handler in root.handlers:
                                handler.close()
                            root.handlers[:] = previous_handlers
                            root.setLevel(previous_level)
                            os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
