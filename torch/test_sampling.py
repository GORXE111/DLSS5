"""CPU regression checks for measured sampler ties, FMA and failed reports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import check_sampling
from dlss5 import ops
from dlss5 import ptx_math


class SamplingTests(unittest.TestCase):
    def test_unknown_cli_option_invalidates_previous_pass_before_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder)/'report.json'
            report.write_text('{"passed":true,"old":true}')
            with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('must remain CPU')):
                with self.assertRaises(SystemExit) as error:
                    check_sampling.main(['--report',str(report),'--invalid-option'])
            self.assertEqual(error.exception.code,2)
            self.assertFalse(json.loads(report.read_text())['passed'])

    def test_reciprocal_cpu_fallback_and_rejected_inputs(self):
        x=torch.tensor([1.,2.,37.,-4.])
        torch.testing.assert_close(ptx_math.rcp_approx(x),x.reciprocal(),rtol=0,atol=0)
        with self.assertRaises(TypeError):ptx_math.rcp_approx(x.double())
        with self.assertRaises(RuntimeError):ptx_math.rcp_approx(x.requires_grad_())

    def test_pre_native_preserves_coordinate_ftz_context(self):
        old=check_sampling.probe_ptx('pre')
        native=check_sampling.probe_ptx('pre_native')
        self.assertIn(b'mul.f32 %r299',old)
        self.assertIn(b'mul.ftz.f32 %r299',native)
        self.assertEqual(old.replace(b'mul.f32 %r299',b'mul.ftz.f32 %r299').replace(b'mul.f32 %r301',b'mul.ftz.f32 %r301'),native)

    def test_explicit_arithmetic_modes_and_invalid_mode(self):
        image=torch.arange(18,dtype=torch.float32).reshape(2,3,3)/18
        u,v=torch.tensor([.25,.75]),torch.tensor([.25,.75])
        torch.testing.assert_close(ops.catmull_rom5(image,u,v),ops.catmull_rom5(image,u,v,arithmetic='fused'),rtol=0,atol=0)
        self.assertTrue(torch.isfinite(ops.catmull_rom5(image,u,v,arithmetic='split')).all())
        with self.assertRaises(ValueError):
            ops.catmull_rom5(image,u,v,arithmetic='unknown')

    def test_report_hardlinks_and_temporary_aliases_preserve_inputs(self):
        for relative in ('torch/dlss5/ops.py','torch/dlss5/net.py','fatbins/source.ptx',
                         'sm86_port/source.cubin','dlss5/model.dll','research/tapspost/nr-trace.tsv',
                         'research/out_f0.ppm','research/validation/batch3_sampling_current.json'):
            for temporary in (False,True):
                with self.subTest(input=relative,temporary=temporary),tempfile.TemporaryDirectory() as folder:
                    root=Path(folder)/'workspace'
                    source=root/relative
                    source.parent.mkdir(parents=True,exist_ok=True)
                    source.write_bytes(b'preserve original reference bytes')
                    report=Path(folder)/'output.json'
                    alias=report.with_suffix('.json.tmp') if temporary else report
                    alias.hardlink_to(source)
                    with patch.object(check_sampling,'ROOT',root),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('must remain CPU')):
                        self.assertEqual(check_sampling.main(['--report',str(report)]),2)
                    self.assertEqual(source.read_bytes(),b'preserve original reference bytes')

    def test_report_inside_source_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'torch').mkdir()
            report=root/'torch/report.json'
            with patch.object(check_sampling,'ROOT',root):
                self.assertEqual(check_sampling.main(['--report',str(report)]),2)
            self.assertFalse(report.exists())

    def test_partial_suite_does_not_claim_complete_pass(self):
        def isolated(report):report['isolated'].extend([dict(passed=True)]*16)
        def full(report):report['full_kernel'].extend([dict(passed=True)]*9)
        def network(report):
            report['network'].extend([dict(passed=True)]*20)
            report['cuda_graph']=dict(passed=True)
        for flag in ('--isolated-only','--network-only'):
            with self.subTest(flag=flag),tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'report.json'
                with patch.object(torch.cuda,'get_device_name',return_value='CPU fixture'),\
                     patch.object(check_sampling,'isolated_checks',side_effect=isolated),\
                     patch.object(check_sampling,'full_kernel_checks',side_effect=full),\
                     patch.object(check_sampling,'network_checks',side_effect=network),\
                     patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('must remain CPU')):
                    self.assertEqual(check_sampling.main(['--report',str(path),flag]),0)
                report=json.loads(path.read_text())
                self.assertFalse(report['complete'])
                self.assertFalse(report['passed'])
                self.assertTrue(report['requested_suite_passed'])

    def test_full_suite_rejects_missing_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json'
            with patch.object(torch.cuda,'get_device_name',return_value='CPU fixture'),\
                 patch.object(check_sampling,'isolated_checks',side_effect=lambda r:r['isolated'].extend([dict(passed=True)]*15)),\
                 patch.object(check_sampling,'full_kernel_checks',side_effect=lambda r:r['full_kernel'].extend([dict(passed=True)]*9)),\
                 patch.object(check_sampling,'network_checks',side_effect=lambda r:r['network'].extend([dict(passed=True)]*20)),\
                 patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('must remain CPU')):
                self.assertEqual(check_sampling.main(['--report',str(path)]),1)
            self.assertFalse(json.loads(path.read_text())['passed'])

    def test_axis_and_cross_halfway_ties(self):
        image=torch.tensor([[[0.,0.,0.],[0.,1.,0.]],[[0.,0.,1.],[1.,1.,1.]]])
        u=torch.tensor((.5+1.5/256)/2)
        output=ops.bilinear(image,u,torch.tensor(.5))
        # Two measured half-up decisions: axis 1.5/256 -> 2/256,
        # then cross (2/256)*(128/256) -> 1/256.
        torch.testing.assert_close(output,torch.tensor([1/256,2/256,.5]),rtol=0,atol=0)

    def test_outside_coordinates_clamp_to_corners(self):
        image=torch.arange(12,dtype=torch.float32).reshape(2,2,3)
        u=torch.tensor([-1.,2.,-1.,2.]);v=torch.tensor([-1.,-1.,2.,2.])
        torch.testing.assert_close(ops.bilinear(image,u,v),image.reshape(4,3),rtol=0,atol=0)

    def test_motion_uses_single_rounding_fma(self):
        u,v=ops.reproject_uv(torch.tensor([-1.]),torch.tensor([0.]),torch.tensor([[3.,0.]]),3,2)
        self.assertEqual(u.item(),2**-25)
        self.assertEqual(v.item(),0)

    def test_stage_errors_replace_an_old_success_report(self):
        for stage in ('device','isolated','full_kernel','network'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as folder:
                report=Path(folder)/'report.json'
                report.write_text('{"passed":true,"old":true}')
                failing=lambda *args:(_ for _ in ()).throw(RuntimeError('fixture '+stage))
                with patch.object(torch.cuda,'get_device_name',side_effect=failing if stage=='device' else None,return_value='CPU fixture'), \
                     patch.object(check_sampling,'isolated_checks',side_effect=failing if stage=='isolated' else None), \
                     patch.object(check_sampling,'full_kernel_checks',side_effect=failing if stage=='full_kernel' else None), \
                     patch.object(check_sampling,'network_checks',side_effect=failing if stage=='network' else None), \
                     patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CPU tests must not use CUDA')):
                    self.assertEqual(check_sampling.main(['--report',str(report)]),1)
                result=json.loads(report.read_text())
                self.assertFalse(result['passed'])
                self.assertIn('fixture '+stage,result['errors'][0])
                self.assertNotIn('old',result)


if __name__=='__main__':
    unittest.main()
