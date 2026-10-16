"""Native installer contracts for a full GLM-5.3 GGUF, without GPU/network."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import maya
from gguf_writer import GGUFWriter


class FullDsaSetup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.first = self.root / "GLM-5.3-IQ4_XS.gguf"
        w = GGUFWriter()
        w.add("general.architecture", "glm-dsa")
        w.add_f32("output_norm.weight", np.ones(4, np.float32))
        w.write(self.first)
        self.pack = self.root / "pack"
        self.pack.mkdir()
        self.profile = self.root / "expert-profile-glm.bin"
        self.profile.touch()
        for ctx in (patch.object(maya, "ROOT", self.root), patch.object(maya, "PROFILE", self.profile),
                    patch.object(maya, "CAL_STORE", self.root / "calibration.json"),
                    patch.object(maya, "say"), patch.object(maya, "step"), patch.object(maya, "ok"),
                    patch.object(maya, "mem_gb", return_value=(512, 500)),
                    patch.object(maya, "saved_calibration", return_value=None)):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.a = SimpleNamespace(env=[], port=8080, host=None, api_key=None, gguf_dir=str(self.root),
                                 parallel=None, selected_draft="none")
        self.pc = {"backend": "cuda", "gpus": [{"index": 0}, {"index": 1}]}

    def config(self, **kwargs):
        p = maya.write_config(self.a, self.pc, {}, self.pack, "IQ4_XS", 4096, self.root, None,
                              ("local", self.first), **kwargs)
        return json.loads(p.read_text())

    def test_discovers_full_model_without_rewriting_architecture(self):
        before = self.first.read_bytes()
        self.assertEqual(maya.local_first(self.root), (self.first.resolve(), None))
        self.assertEqual(maya.glm_files(self.root), [self.first])
        self.assertEqual(self.first.read_bytes(), before)

    def test_full_config_uses_two_gpus_without_flash_auxiliaries(self):
        cfg = self.config()
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["model_name"], "glm-5.3")
        self.assertEqual(cfg["architecture"], "glm-dsa")
        self.assertEqual(cfg["installer"]["draft"], "none")
        for flag, value in (("--spec", "1"), ("--suffix-draft", "0"),
                            ("--parallel", "1"), ("--kv-streaming", "off"), ("--prefill", "0")):
            self.assertEqual(maya.flag_value(cfg["args"], flag), value)
        self.assertEqual(cfg["env"]["STRATA_GLM_NO_MTP"], "1")
        self.assertEqual(cfg["env"]["STRATA_GLM_KV_INT8"], "0")
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--expert-profile", cfg["args"])
        self.assertFalse(maya.add_profile(cfg))

    def test_reinstall_replaces_unsupported_old_streaming_and_parallel(self):
        cfg = self.config()
        cfg["args"] += ["--kv-resident", "512"]
        maya.set_arg(cfg["args"], "--parallel", 4)
        maya.set_arg(cfg["args"], "--kv-streaming", "on")
        (self.root / "maya-iq4_xs.json").write_text(json.dumps(cfg))
        new = self.config()
        self.assertEqual(maya.flag_value(new["args"], "--parallel"), "1")
        self.assertEqual(maya.flag_value(new["args"], "--kv-streaming"), "off")

    def test_explicit_parallel_and_dflash_rejected(self):
        self.a.parallel = 2
        with self.assertRaises(SystemExit):
            self.config()
        self.a.parallel = 1
        with self.assertRaises(SystemExit):
            self.config(dflash=self.root / "Flash-dflash.gguf")

    def test_none_is_an_explicit_draft_choice(self):
        self.assertEqual(maya.choose_draft(SimpleNamespace(draft="none"), self.pc, {}), "none")


if __name__ == "__main__":
    unittest.main()
