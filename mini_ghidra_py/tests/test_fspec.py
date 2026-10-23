import unittest
from minighidra.fspec import CallSpecs
from minighidra.paramid import PcodeLikeGraph, PcodeOp, ParamActive, ParamRank
from minighidra.paramlist import ProtoModelMerged, PrototypeModel, ParamListStandard, ParamListStandardOut

class TestFspec(unittest.TestCase):
    def test_call_specs(self):
        op = PcodeOp("CALL", (1, 2), output=3)
        specs = CallSpecs(op, name="test_func")
        self.assertEqual(specs.name, "test_func")
        self.assertFalse(specs.is_input_active)
        self.assertFalse(specs.is_output_active)
        
        specs.set_input_active()
        self.assertTrue(specs.is_input_active)
        
        graph = PcodeLikeGraph()
        specs.check_input_trial_use(graph)
        
        # Test mock methods
        specs.resolve_model()
        specs.derive_input_map()
        specs.build_input_from_trials(graph)
        
if __name__ == "__main__":
    unittest.main()
