import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "ops" / "blender_render_ready3d_v2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blender_render_ready3d_v2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BlenderRenderEngineTests(unittest.TestCase):
    def test_headless_render_engine_falls_back_to_eevee_when_cycles_is_unavailable(self):
        module = load_module()

        selected = module.choose_render_engine(
            {"BLENDER_EEVEE", "BLENDER_WORKBENCH"},
            headless=True,
        )

        self.assertEqual(selected, "BLENDER_EEVEE")


if __name__ == "__main__":
    unittest.main()
