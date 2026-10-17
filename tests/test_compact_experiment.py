import unittest
from bench.experiments.compact_live_experiment import command_recipe


class CompactTests(unittest.TestCase):
    def test_fill_binds_literal_without_interpreting_delimiters(self):
        text=r'Read then write; C:\notes'
        plan=command_recipe([{'op':'fill','target':'Notes','text':text},{'op':'click','target':'Save'}])
        self.assertEqual(len(plan['steps']),2)
        self.assertEqual(plan['steps'][0]['actions'][-1]['text'],text)

    def test_rejects_invented_ops_and_commands_in_targets(self):
        for cmd in [{'op':'eval','target':'Notes'}, {'op':'click','target':'Save; click Delete'},
                    {'op':'fill','target':'Name','text':'café'}]:
            with self.subTest(cmd=cmd):self.assertIsNone(command_recipe([cmd]))

    def test_empty_literal_really_clears_the_field(self):
        plan=command_recipe([{'op':'fill','target':'Name','text':''}])
        self.assertEqual(plan['steps'][0]['actions'][-1],{'type':'hotkey','keys':['BACKSPACE']})

    def test_drag_preserves_both_arguments(self):
        plan=command_recipe([{'op':'drag','target':'Report','destination':'Archive'}])
        self.assertEqual(plan['steps'][0]['actions'][0]['destination'],'Archive')


if __name__=='__main__':unittest.main()
