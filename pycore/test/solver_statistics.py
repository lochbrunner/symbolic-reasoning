#!/usr/bin/env python3

import unittest

import os
import shutil
import tempfile

from pycore import StepInfo, TraceStatistics, ProblemStatistics, SolverStatistics, ProblemSummary, IterationSummary


class TestStepInfo(unittest.TestCase):
    def test_simple_attributes(self):
        step = StepInfo()

        step.current_latex = 'a+b'
        self.assertEqual(step.current_latex, 'a+b')

        step.value = 0.7
        self.assertAlmostEqual(step.value, 0.7)

        step.confidence = 0.2
        self.assertAlmostEqual(step.confidence, 0.2)

        step.rule_id = 4
        self.assertEqual(step.rule_id, 4)

        step.path = [0, 1]
        self.assertEqual(step.path, [0, 1])

        step.top = 1
        self.assertEqual(step.top, 1)

        step.contributed = True
        self.assertEqual(step.contributed, True)

    def test_subsequent(self):
        root = StepInfo()

        self.assertEqual(root.subsequent, [])

        c = StepInfo()
        c.current_latex = 'c'
        root.add_subsequent(c)
        self.assertEqual(root.subsequent, [c])

        d = StepInfo()
        d.current_latex = 'd'
        root += d
        self.assertEqual(root.subsequent, [c, d])


class TestTraceStatistics(unittest.TestCase):
    def test_small(self):
        step = StepInfo()
        step.current_latex = 'a+b'

        trace = TraceStatistics()

        self.assertEqual(trace.success, False)
        trace.success = True
        self.assertEqual(trace.success, True)

        self.assertEqual(trace.fit_tries, 0)
        trace.fit_tries = 12
        self.assertEqual(trace.fit_tries, 12)

        self.assertEqual(trace.fit_results, 0)
        trace.fit_results = 25
        self.assertEqual(trace.fit_results, 25)

        trace.trace = step
        self.assertEqual(trace.trace, step)


class TestProblemStatistics(unittest.TestCase):
    def test_naming(self):
        problem = ProblemStatistics('Problem a', 'a')
        self.assertEqual(problem.problem_name, 'Problem a')

        problem.problem_name = 'Problem b'
        self.assertEqual(problem.problem_name, 'Problem b')

    def test_target_latex(self):
        problem = ProblemStatistics('Problem a', 'a')
        self.assertEqual(problem.target_latex, 'a')
        problem.target_latex = 'b'
        self.assertEqual(problem.target_latex, 'b')

    def test_iterations(self):
        problem = ProblemStatistics('Problem a', 'a')
        self.assertEqual(problem.iterations, [])

        trace = TraceStatistics()
        problem.add_iteration(trace)
        self.assertEqual(len(problem.iterations), 1)

        trace = TraceStatistics()
        problem += trace
        self.assertEqual(len(problem.iterations), 2)


class TestSolverStatistics(unittest.TestCase):
    def test_construction(self):
        stats = SolverStatistics()

        self.assertEqual(stats.header, [])

        problem = ProblemStatistics('Problem a', 'a')
        stats += problem

        expected_summary = ProblemSummary('Problem a', False, target_latex='a')

        self.assertEqual(stats.header[0], expected_summary)

    def test_io(self):
        stats = SolverStatistics()
        trace = TraceStatistics()
        problem = ProblemStatistics('Problem a', 'a')
        problem += trace
        stats += problem

        directory = tempfile.mkdtemp()
        try:
            filename = os.path.join(directory, 'test.spb')
            stats.dump(filename)

            loaded = SolverStatistics.load(filename)

            expected_summary = ProblemSummary('Problem a', False, [IterationSummary(0, False, 1)], '', target_latex='a')
            self.assertEqual(loaded.header[0], expected_summary)

        finally:
            shutil.rmtree(directory)


if __name__ == '__main__':
    unittest.main()
