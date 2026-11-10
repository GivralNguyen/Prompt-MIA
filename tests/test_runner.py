"""Check experiment coverage, dependency reuse, and interrupted-job handling."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from promptmia.io import project_root, read_json, write_json
from promptmia.workflow import build_plan, execute, load_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(project_root() / "configs" / "paper.json")

    def test_exact_paper_outputs_and_no_unused_experiments(self):
        expected = {f"figure{i}" for i in range(1, 22)} | {f"table{i}" for i in range(1, 10)}
        self.assertEqual({a["id"] for a in self.manifest["artifacts"]}, expected)
        plan = build_plan(self.manifest, self.manifest["artifacts"])
        for section in ("training", "evaluation"):
            self.assertEqual({j["name"] for j in plan[section]}, {j["name"] for j in self.manifest[section]})

    def test_shared_training_is_resolved_once(self):
        outputs = [a for a in self.manifest['artifacts'] if a['id'] in ('figure4', 'table1')]
        plan = build_plan(self.manifest, outputs)
        self.assertEqual(len(plan["training"]), 12)
        self.assertEqual(len(plan["evaluation"]), 21)
        self.assertEqual(len(plan["artifacts"]), 2)

    def test_paper_panel_settings(self):
        artifacts = {a["id"]: a for a in self.manifest["artifacts"]}
        self.assertEqual(len(artifacts["figure17"]["panels"]), 6)
        self.assertEqual(len(artifacts["figure20"]["panels"]), 3)
        self.assertEqual(len(artifacts["figure10"]["panels"]), 2)
        plan = build_plan(self.manifest, [artifacts["figure19"]])
        self.assertEqual({j["backbone"] for j in plan["training"]}, {"vit_b32", "deit_b16", "convit_base"})
        self.assertEqual(len(plan["training"]), 12)
        for panel in artifacts["figure19"]["panels"]:
            self.assertEqual(set(panel["metrics"]), {"asr", "advantage"})

    def test_each_output_has_an_independent_script_and_complete_configuration(self):
        root = project_root()
        self.assertFalse((root / 'reproduce.py').exists())
        for artifact in self.manifest['artifacts']:
            name = artifact['id']
            self.assertTrue((root / 'scripts' / (name.replace('figure', 'fig') + '.py')).is_file())
            config = load_manifest(root / 'configs' / (name + '.json'))
            expected = build_plan(self.manifest, [artifact])
            self.assertEqual(config['artifacts'], [artifact])
            self.assertEqual(config['training'], expected['training'])
            self.assertEqual(config['evaluation'], expected['evaluation'])

    def test_key_learning_rates_and_federation_settings(self):
        artifacts = {a['id']: a for a in self.manifest['artifacts']}
        main = build_plan(self.manifest, [artifacts['figure4']])
        self.assertTrue(all(j['key_lr'] == .001 and j['lr'] == .0001 for j in main['training']))
        rounds = build_plan(self.manifest, [artifacts['figure17']])['training']
        self.assertTrue(all(j['key_lr'] == .0001 for j in rounds if j['name'].startswith('dynamics/')))
        dynamics = build_plan(self.manifest, [artifacts['figure18']])
        self.assertTrue(all(j['key_lr'] == .001 for j in dynamics['training']))
        self.assertEqual({j['aggregator'] for j in self.manifest['training']}, {'fedavg'})
        clients = build_plan(self.manifest, [artifacts['figure21']])
        self.assertEqual({j['clients'] for j in clients['training']}, {80, 160, 320})
        self.assertEqual({j['dataset'] for j in clients['training']},
                         {'cifar10', 'cifar100', 'tinyimagenet', 'fourdataset'})

    def test_missing_and_cyclic_dependencies_are_errors(self):
        with self.assertRaisesRegex(ValueError, "Missing evaluation"):
            build_plan(dict(training=[], evaluation=[]), [dict(sources=["unknown"])])
        manifest = dict(training=[], evaluation=[dict(name="a", depends_on=["b"]),
                                                  dict(name="b", depends_on=["a"])])
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            build_plan(manifest, [dict(sources=["a"])])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calls = dict(training=0, evaluation=0, artifacts=0)
        self.fail_training = False
        self.plan = dict(training=[dict(name="tiny", lr=.01)],
                         evaluation=[dict(name="game", kind="attack", run="tiny")],
                         artifacts=[dict(id="figure4", sources=["game"])])

        def train(job, root, device):
            self.calls["training"] += 1
            if self.fail_training:
                raise RuntimeError("interrupted training")
            run = root / "runs" / job["name"]
            write_json(run / "config.json", dict(job, query_cache="cache/queries/tiny.pt"))
            write_json(run / "metrics.json", dict(completed=True))
            for path in (run / "keys.pt", run / "state.pt", root / "cache/queries/tiny.pt"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(self.calls["training"]))

        def evaluate(job, root, device):
            self.calls["evaluation"] += 1
            result = dict(asr=.75, iteration=self.calls["evaluation"])
            write_json(root / "results" / (job["name"] + ".json"), result)
            return result

        def plot(job, root):
            self.calls["artifacts"] += 1
            output = root / "figures" / (job["id"] + ".json")
            write_json(output, dict(iteration=self.calls["artifacts"]))
            return [output]

        modules = {}
        for module, name, function in (("training", "train", train), ("evaluation", "evaluate_job", evaluate),
                                       ("plotting", "plot_artifact", plot)):
            stub = types.ModuleType("promptmia." + module)
            setattr(stub, name, function)
            modules[stub.__name__] = stub
        self.patcher = patch.dict("sys.modules", modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        code = patch("promptmia.workflow.code_fingerprint", return_value="unchanged-code")
        code.start()
        self.addCleanup(code.stop)
        # Keep environment changes made by this isolated runner out of other tests.
        environment = patch.dict("os.environ")
        environment.start()
        self.addCleanup(environment.stop)

    def test_completed_jobs_are_reused_and_changed_inputs_redraw(self):
        execute(self.plan, self.root, "all", "cpu")
        execute(self.plan, self.root, "all", "cpu")
        self.assertEqual(self.calls, dict(training=1, evaluation=1, artifacts=1))
        write_json(self.root / "results/game.json", dict(asr=.8, changed=True))
        execute(self.plan, self.root, "plot", "cpu")
        self.assertEqual(self.calls["artifacts"], 2)
        self.plan["training"][0]["lr"] = .02
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            execute(self.plan, self.root, "evaluate", "cpu")
        execute(self.plan, self.root, "all", "cpu")
        self.assertEqual(self.calls, dict(training=2, evaluation=2, artifacts=3))

    def test_failed_forced_rerun_removes_success_record(self):
        execute(self.plan, self.root, "train", "cpu")
        stamp = self.root / "logs/completed/training/tiny.json"
        self.assertTrue(stamp.exists())
        self.fail_training = True
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            execute(self.plan, self.root, "train", "cpu", force=True)
        self.assertFalse(stamp.exists())
        self.fail_training = False
        execute(self.plan, self.root, "train", "cpu")
        self.assertEqual(self.calls["training"], 3)

    def test_incomplete_training_cannot_be_evaluated(self):
        execute(self.plan, self.root, "train", "cpu")
        write_json(self.root / "runs/tiny/metrics.json", dict(completed=False))
        with self.assertRaisesRegex(ValueError, "not completed"):
            execute(self.plan, self.root, "evaluate", "cpu")


if __name__ == "__main__":
    unittest.main()
