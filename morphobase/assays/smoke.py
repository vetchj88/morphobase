from morphobase.assays.common import AssayResult, AssayRunner, rollout_body


class SmokeAssay(AssayRunner):
    def run(self, cfg):
        rollout = rollout_body(cfg)
        return AssayResult(
            history=rollout["history"],
            final_metrics=rollout["final_metrics"],
            notes="Synthetic smoke assay completed.",
        )
