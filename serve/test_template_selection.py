"""Architecture, rather than template resemblance, decides the Flash override."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from serve import server
from serve.frontend import ChatTemplate


class TemplateSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pack = self.root / "tokenizer"
        self.pack.mkdir()
        (self.root / "serve").mkdir()
        self.own = self.pack / "chat_template.jinja"
        # This is precisely upstream's heuristic; it cannot distinguish full
        # from Flash. Distinct output proves the selected file is actually used.
        self.own.write_text("[gMASK]<sop>{% if reasoning_effort in ['low', 'high'] %}pack-low{% else %}pack-max{% endif %}")
        self.shipped = self.root / "serve/glm5_chat_template.jinja"
        self.shipped.write_text("shipped-flash")
        self.fallback = self.root / "serve/chat_template.jinja"
        self.fallback.write_text("fallback")
        for ctx in (patch.object(server, "ROOT", self.root), patch.dict(os.environ, {}, clear=True)):
            ctx.start()
            self.addCleanup(ctx.stop)

    def test_full_keeps_pack_template_despite_flash_heuristic(self):
        chosen = server.chat_template_path(self.pack, "glm-dsa")
        self.assertEqual(chosen, self.own)
        self.assertEqual(ChatTemplate(chosen).render([], reasoning_effort="high"), "[gMASK]<sop>pack-low")

    def test_native_info_selects_full_template_without_installer_config(self):
        fake = self.root / "native-info.py"
        fake.write_text("#!/usr/bin/env python3\nimport sys\n"
                        "print('INFO architecture=glm-dsa', flush=True)\n"
                        "print('READY 4096 stop', flush=True)\n"
                        "for line in sys.stdin:\n"
                        "    if line.strip() == 'QUIT': break\n")
        fake.chmod(0o755)
        engine = server.StrataEngine(str(fake), [])
        try:
            self.assertEqual(engine.info["architecture"], "glm-dsa")
            self.assertEqual(server.chat_template_path(self.pack, engine.info["architecture"]), self.own)
        finally:
            engine.close()
            engine.proc.stdin.close()
            engine.proc.stdout.close()

    def test_flash_keeps_upstream_override(self):
        chosen = server.chat_template_path(self.pack, "glm5-next")
        self.assertEqual(chosen, self.shipped)
        self.assertEqual(ChatTemplate(chosen).render([]), "shipped-flash")

    def test_unidentified_existing_flash_config_keeps_upstream_heuristic(self):
        self.assertEqual(server.chat_template_path(self.pack), self.shipped)

    def test_pack_override_still_applies_to_flash(self):
        os.environ["MAYA_PACK_TEMPLATE"] = "1"
        self.assertEqual(server.chat_template_path(self.pack, "glm5-next"), self.own)

    def test_other_template_and_missing_pack_keep_existing_fallbacks(self):
        self.own.write_text("another model")
        self.assertEqual(server.chat_template_path(self.pack, "qwen4exp"), self.own)
        self.own.unlink()
        self.assertEqual(server.chat_template_path(self.pack, "glm-dsa"), self.fallback)


if __name__ == "__main__":
    unittest.main()
