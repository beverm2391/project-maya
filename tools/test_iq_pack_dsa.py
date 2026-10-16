"""GLM-DSA packing over real split GGUF files; no GPU or model download."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py
add_gguf_py()
from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
from gguf_reader import GGUFFile
import iq_pack


def shard(path, tensors, pad=0):
    w = GGUFWriter(path, "glm-dsa")
    w.add_uint32("glm-dsa.block_count", 3)
    w.add_uint32("glm-dsa.nextn_predict_layers", 1)
    w.add_uint32("glm-dsa.leading_dense_block_count", 1)
    if pad:
        w.add_string("general.name", "x" * pad)
    for name, values, kind in tensors:
        # IQ bytes are already quantized: preserve their blocks byte for byte.
        raw = values if values.dtype == np.uint8 else quants.quantize(values, kind)
        w.add_tensor(name, raw, raw_dtype=kind)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class DsaPackTests(unittest.TestCase):
    def test_architecture_precision_contract(self):
        for name in iq_pack.GLMDSA_BF16:
            self.assertTrue(iq_pack.needs_bf16("blk.3." + name, "Q8_0", "glm-dsa"))
        for name in ("hc_attn_fn.weight", "ssm_beta.weight", "indexer_compressor_gate.weight"):
            self.assertFalse(iq_pack.needs_bf16("blk.3." + name, "Q8_0", "glm-dsa"))
            self.assertTrue(iq_pack.needs_bf16("blk.3." + name, "Q8_0", "glm5-next"))
        self.assertFalse(set(iq_pack.GLMDSA_BF16) & set(iq_pack.GLMDSA_NATIVE_OK))

    def test_pinned_warlock_header_signatures_have_a_loader_path(self):
        # All 21 headers at revision 6a0282dd0b39a36d125060933c8d30825d90b602
        # were inspected by range read. These are the observed nonexpert stems;
        # this check uses that evidence without downloading any tensor payload.
        q8 = ("attn_k_b.weight", "attn_v_b.weight", "attn_kv_a_mqa.weight",
              "attn_output.weight", "attn_q_a.weight", "attn_q_b.weight",
              "ffn_down.weight", "ffn_gate.weight", "ffn_up.weight",
              "ffn_down_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
              "indexer.attn_k.weight", "indexer.attn_q_b.weight")
        f32 = ("attn_kv_a_norm.weight", "attn_q_a_norm.weight", "attn_norm.weight", "ffn_norm.weight",
               "ffn_gate_inp.weight", "exp_probs_b.bias", "indexer.k_norm.weight",
               "indexer.k_norm.bias", "indexer.proj.weight")
        for stem in q8:
            name = "blk.3." + stem
            self.assertTrue(iq_pack.needs_bf16(name, "Q8_0", "glm-dsa") or
                            name.endswith(iq_pack.GLMDSA_NATIVE_OK), name)
        for stem in f32:
            self.assertNotIn(stem, iq_pack.GLMDSA_NATIVE_OK)
        # Exact real MLA shapes preserve their GGUF byte order in the flat index.
        self.assertEqual(iq_pack.index_shape((192, 512, 64)), (98304, 64))
        self.assertEqual(iq_pack.index_shape((512, 256, 64)), (131072, 64))

    def test_split_iq4_experts_trunk_and_mtp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / ("full-%05d-of-00003.gguf" % i) for i in range(1, 4)]
            f = np.ones((2, 256), np.float32)
            # Two experts, 256 x 256 each; every IQ4_XS row has 136 bytes.
            roles = [np.full((2, 256, 136), i, np.uint8) for i in (1, 2, 3)]
            shard(paths[0], [("blk.0.ffn_gate.weight", f, Q.Q8_0),
                             ("blk.1.ffn_gate_inp.weight", f, Q.F32),
                             ("blk.1.attn_k_b.weight", np.ones((2, 2, 256), np.float32), Q.BF16),
                             ("blk.1.ffn_gate_exps.weight", roles[0], Q.IQ4_XS)])
            shard(paths[1], [("blk.1.ffn_up_exps.weight", roles[1], Q.IQ4_XS)], pad=4096)
            shard(paths[2], [("blk.1.ffn_down_exps.weight", roles[2], Q.IQ4_XS),
                             ("blk.2.nextn.eh_proj.weight", np.zeros((2, 144), np.uint8), Q.Q4_K),
                             ("blk.2.ffn_gate_inp.weight", f, Q.F32)] +
                            [("blk.2.ffn_%s_exps.weight" % r, np.zeros((2, 256, 144), np.uint8), Q.Q4_K)
                             for r in iq_pack.ROLES], pad=128)
            out = root / "pack"
            (out / "tokenizer").mkdir(parents=True)
            for name in ("vocab.json", "chat_template.jinja"):
                (out / "tokenizer" / name).touch()
            before = [p.read_bytes() for p in paths]
            with patch.object(sys, "argv", ["iq_pack", "--gguf", str(paths[0]), "--out", str(out),
                                           "--compat-bf16"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.main(), 0)
            _, index = iq_pack.read_index(out / "index.txt")
            self.assertFalse(any(n.startswith("blk.2.") for n in index))
            self.assertEqual(index["blk.0.ffn_gate.weight"][9], "8")
            self.assertEqual(index["blk.1.ffn_gate_inp.weight"][2], "2")
            self.assertEqual(index["blk.1.attn_k_b.weight"][7:9], ["512", "2"])
            lines = (out / "native_experts.txt").read_text().splitlines()
            self.assertIn("v4", lines[0])
            self.assertIn("n_expert 2,", lines[0])
            row = lines[1].split()
            self.assertEqual(row[:3], ["1", "23", "23"])
            self.assertEqual(row[8:], [p.name for p in paths])
            blob = int(row[4])
            self.assertEqual(blob, 3 * 256 * 136)
            for i, p in enumerate(paths):
                g = GGUFFile(p)
                t = next(t for t in g.tensors if t.name == "blk.1.ffn_%s_exps.weight" % iq_pack.ROLES[i])
                self.assertEqual(int(row[5 + i]), g.data_start + t.offset)
                with p.open("rb") as f:
                    f.seek(int(row[5 + i]))
                    self.assertEqual(f.read(t.expected_bytes()), roles[i].tobytes())
            self.assertEqual([p.read_bytes() for p in paths], before)
            self.assertFalse((out / "experts.bin").exists())

    def test_only_full_indexer_and_absorbed_projections_convert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "model.gguf"
            values = np.linspace(-1, 1, 512, dtype=np.float32).reshape(2, 256)
            names = ["blk.1." + n for n in iq_pack.GLMDSA_BF16]
            shard(p, [(n, values, Q.Q8_0) for n in names] + [
                ("blk.1.ffn_gate_inp.weight", values, Q.F32),
                ("blk.1.indexer.k_norm.bias", np.ones(128, np.float32), Q.F32),
                ("blk.2.attn_k_b.weight", values, Q.Q8_0)])
            model = iq_pack.Model(p)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(p, root, model, True, "glm-dsa", 2), 0)
            _, rows = iq_pack.read_index(root / "index.txt")
            dense = (root / "dense.bin").read_bytes()
            expected = iq_pack.bf16_bytes(quants.quantize(values, Q.Q8_0), "Q8_0")
            for n in names:
                row = rows[n]
                self.assertEqual(row[2], "4")
                self.assertEqual(dense[int(row[3]):int(row[3]) + int(row[4])], expected)
            self.assertEqual(rows["blk.1.ffn_gate_inp.weight"][2], "2")
            self.assertEqual(rows["blk.1.indexer.k_norm.bias"][2], "2")
            self.assertNotIn("blk.2.attn_k_b.weight", rows)

    def test_unknown_quantized_dense_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "model.gguf"
            shard(p, [("blk.0.hc_attn_fn.weight", np.ones((2, 256), np.float32), Q.Q8_0)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(p, root, iq_pack.Model(p), True, "glm-dsa", 2), 1)


if __name__ == "__main__":
    unittest.main()
