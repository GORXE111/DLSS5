"""CPU resource-presence contracts, without weights or GPU initialization."""
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dlss5 import net as N
from dlss5.export import Core


def pre_block(device="cpu"):
    pre = N.PreBlock.__new__(N.PreBlock)
    pre.dev, pre._kc = device, {}
    return pre


def core_parts(height, width, device="cpu"):
    pre = pre_block(device)
    post = N.PostBlock.__new__(N.PostBlock)
    post.dev, post.scale, post.blend = device, .03125, .75
    gh, gw = N.dims(height, width)["grid"]
    n = torch.zeros(gh, gw, 4, device=device)
    post.net = lambda *args: n
    core = Core.__new__(Core)
    torch.nn.Module.__init__(core)
    core.net = SimpleNamespace(steps=[({}, pre), ({}, post)])
    core.canon = False
    return pre, post, core, n


class ResetTests(unittest.TestCase):
    def test_pre_rgb_half_boundaries_in_both_modes_and_autocast(self):
        # Exact half bit patterns at input-conversion ties and SUB.F16 boundaries.
        values=torch.tensor([0.,1.,2**-14,2**-10,.0001,.4998779296875,.4998778998851776,
                             .500244140625,.5002442002296448,1.00048828125,1.0004884004592896,-.0001])
        bits=[0xac00,0x2c00,0xac00,0xabfc,0xac00,0,0x8200,0,0x0400,0x2c00,0x2c02,0xac00]
        expected=torch.tensor([v-65536 if v>=32768 else v for v in bits],dtype=torch.int16)
        for precise in (False,True):
            for autocast in (False,True):
                with self.subTest(precise=precise,autocast=autocast),patch.object(N.ops,'PRECISE',precise), \
                     torch.autocast('cpu',dtype=torch.bfloat16,enabled=autocast):
                    output=N._pre_rgb(values)
                    self.assertEqual(output.dtype,torch.float32)
                    self.assertTrue(torch.equal(output.half().view(torch.int16),expected))

    def test_pre_rgb_rounding_leaves_noise_and_controls_float32(self):
        h,w=8,16;gh,gw=N.dims(h,w)['grid']
        color=torch.full((h,w,3),.5002442002296448)
        block=pre_block()
        nz=torch.tensor([.123456789,-.987654321,1.00012345])[:,None,None].expand(3,gh,gw)
        ctrl=N.control_inputs(tone=1.234567,structure=.812345,skin=.345678)
        with patch.object(N,'noise',return_value=nz):
            inputs=block.inputs(color,None,None,0,ctrl)
        self.assertTrue(torch.equal(inputs[:,6:9],nz.permute(1,2,0).reshape(-1,3)))
        key=torch.tensor((1.,)+tuple(ctrl[name] for name in block.CONTROLS))
        self.assertTrue(torch.equal(inputs[:,9:],key.expand(gh*gw,-1)))
        self.assertTrue(torch.equal(inputs[:,:3],inputs[:,3:6]))

    def test_pre_and_core_use_color_if_either_resource_missing(self):
        for h, w in ((8, 16), (37, 53)):
            pre, _, core, _ = core_parts(h, w)
            color = torch.rand(h, w, 3)
            hist = 1-color
            mv = torch.empty(h, w, 2)
            mv[..., 0], mv[..., 1] = -3.25, 1.75
            for history, motion in ((None, None), (None, mv), (hist, None)):
                with self.subTest(size=(h,w), history=history is not None, motion=motion is not None):
                    actual = pre.inputs(color, history, motion, 3)
                    self.assertTrue(torch.equal(actual[:, :3], actual[:, 3:6]))
                    self.assertTrue(torch.equal(core.inputs(color, history, motion, 3)[:, :15], actual.half()))

    def test_post_and_core_disable_blend_if_either_resource_missing(self):
        h,w=37,53
        _,post,core,n=core_parts(h,w)
        color=torch.rand(h,w,3)
        hist=1-color
        motion=torch.full((h,w,2),3.25)
        for history,mv in ((None,None),(None,motion),(hist,None)):
            with self.subTest(history=history is not None,motion=mv is not None):
                self.assertTrue(torch.equal(post(None,None,color,history,mv),color))
                self.assertTrue(torch.equal(core.finish(n,color,history,mv),color))
        self.assertFalse(torch.equal(post(None,None,color,hist,motion),color))
        self.assertTrue(torch.equal(post(None,None,color,hist,motion),core.finish(n,color,hist,motion)))

    def test_public_forward_preserves_zero_motion_default_and_missing_history(self):
        model=N.DLSS5.__new__(N.DLSS5)
        model.dev="cpu"
        seen=[]
        def pre(color,hist,mv,frame,controls):
            seen.append((hist,mv))
            return color,color
        def post(x,skip,color,hist,mv):
            seen.append((hist,mv))
            return color
        model.steps=[({"kind":"pre"},pre),({"kind":"post"},post)]
        color=torch.rand(8,16,3)
        mv=torch.full((8,16,2),-1.5)
        model._forward(color,None,mv,0,None,None)
        self.assertIsNone(seen[0][0])
        self.assertIsNone(seen[1][0])
        seen.clear()
        model._forward(color,color,None,1,None,None)
        for hist,motion in seen:
            self.assertIs(hist,color)
            self.assertEqual(torch.count_nonzero(motion).item(),0)

    def test_reset_graph_rejects_history_and_ignores_motion(self):
        # Capture machinery is mocked; the closure's state validation/copy path is real.
        import contextlib
        class Stream:
            def wait_stream(self,*args): pass
        class Graph:
            def replay(self): pass
        model=N.DLSS5.__new__(N.DLSS5)
        model.dev="cpu"
        seen_motion=[]
        def capture_call(self,color,hist,motion,frame,**kwargs):
            seen_motion.append(motion)
            return color
        with patch.object(N.DLSS5,"__call__",capture_call), \
             patch.object(torch.cuda,"Stream",Stream), \
             patch.object(torch.cuda,"current_stream",Stream), \
             patch.object(torch.cuda,"stream",lambda *args:contextlib.nullcontext()), \
             patch.object(torch.cuda,"CUDAGraph",Graph), \
             patch.object(torch.cuda,"graph",lambda *args:contextlib.nullcontext()):
            reset=model.graph(8,16,history=False)
            color=torch.rand(8,16,3)
            self.assertTrue(torch.equal(reset(color,mv=torch.ones(8,16,2)),color))
            self.assertEqual(torch.count_nonzero(seen_motion[-1]).item(),0)
            with self.assertRaisesRegex(ValueError,"fixed history"):
                reset(color,hist=color)
            history=model.graph(8,16,history=True)
            with self.assertRaisesRegex(ValueError,"fixed history"):
                history(color)

    def test_checker_failure_invalidates_old_pass_without_gpu(self):
        import check_reset
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'report.json'
            report.write_text('{"passed":true}')
            with patch.object(check_reset,'baseline_cpu',side_effect=RuntimeError('CPU failure fixture')), \
                 patch.object(torch.cuda,'get_device_name',side_effect=AssertionError('must not initialize GPU')):
                self.assertEqual(check_reset.main(['--report',str(report),'--capture',str(Path(directory)/'capture')]),1)
            current=json.loads(report.read_text())
            self.assertFalse(current['passed'])
            self.assertIn('CPU failure fixture',current['errors'][0])

    def test_checker_rejects_report_alias_without_destroying_input(self):
        import check_reset
        with tempfile.TemporaryDirectory() as directory:
            capture=Path(directory)/'capture';capture.mkdir()
            manifest=capture/'manifest.json';manifest.write_bytes(b'original evidence')
            alias=Path(directory)/'outside.json';alias.hardlink_to(manifest)
            with self.assertRaisesRegex(ValueError,'aliases'):
                check_reset.safe_report(alias,capture)
            self.assertEqual(manifest.read_bytes(),b'original evidence')

    def test_unknown_cli_invalidates_old_pass_before_exit(self):
        import check_reset
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'report.json';report.write_text('{"passed":true}')
            with self.assertRaises(SystemExit) as error:
                check_reset.main(['--report',str(report),'--capture',str(Path(directory)/'capture'),'--invalid-option'])
            self.assertEqual(error.exception.code,2)
            value=json.loads(report.read_text())
            self.assertFalse(value['passed'])
            self.assertFalse(value['suite_complete'])
            self.assertTrue(value['errors'])


if __name__=="__main__":
    torch.set_num_threads(4)
    unittest.main()
