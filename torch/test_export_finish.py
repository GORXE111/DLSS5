"""Keep the engine's Torch finish aligned with the normal temporal finish."""
from types import SimpleNamespace
import unittest

import torch

from dlss5.export import Core
from dlss5.net import PostBlock, dims


class FinishTests(unittest.TestCase):
    def test_engine_finish_matches_post_for_reset_and_motion(self):
        generator = torch.Generator().manual_seed(3108)
        for height, width in ((8, 16), (37, 53)):
            gh, gw = dims(height, width)["grid"]
            color = torch.rand(height, width, 3, generator=generator)
            history = torch.rand(height, width, 3, generator=generator)
            n = torch.randn(gh, gw, 4, generator=generator)
            post = PostBlock.__new__(PostBlock)
            post.dev, post.scale, post.blend = "cpu", 0.03125, 0.75
            post.net = lambda *args: n
            core = Core.__new__(Core)
            torch.nn.Module.__init__(core)
            core.net = SimpleNamespace(steps=[(None, post)])
            for hist in (None, history):
                for dx, dy in ((0, 0), (3, 2), (-3.5, 1.75), (80, -60)):
                    with self.subTest(size=(height, width), reset=hist is None, motion=(dx, dy)):
                        motion = torch.empty(height, width, 2)
                        motion[..., 0], motion[..., 1] = dx, dy
                        expected = post(None, None, color, hist, motion)
                        actual = core.finish(n, color, hist, motion)
                        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
