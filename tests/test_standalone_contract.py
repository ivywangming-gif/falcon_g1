"""Stdlib-only contract test entry point for the standalone audit round."""

import unittest

from falcon_g1 import ActionSplit, GroundContactResetContract, StandaloneTaskPlan


class StandaloneContractTests(unittest.TestCase):
    def test_action_split(self):
        split = ActionSplit()
        self.assertEqual(split.lower_body + split.upper_body, split.total_dofs)

    def test_reset_contract(self):
        GroundContactResetContract().validate_root_state(13)

    def test_plan_defers_runtime(self):
        plan = StandaloneTaskPlan()
        plan.validate()
        self.assertFalse(plan.ppo_enabled)


if __name__ == "__main__":
    unittest.main()
