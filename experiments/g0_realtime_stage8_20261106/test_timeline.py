import copy
import unittest
from experiments.g0_realtime_stage7_20261106.runtime import Runtime as V7
from .runtime import Runtime
from .timeline import analyze


def record(i, admitted, network, post, released):
    return dict(frame_index=i,wall_intervals=dict(admitted=admitted,network=network,post=post,released=released))


class TimelineTests(unittest.TestCase):
    def test_runtime_is_exact_v7_and_serial_has_no_overlap(self):
        self.assertIs(Runtime, V7)
        rows = [record(i,i,(i+.1,i+.3),(i+.4,i+.8),i+.9) for i in range(3)]
        result = analyze(rows,1)
        self.assertEqual(result['observed_in_flight_peak'],1)
        self.assertEqual(result['overlap_seconds'],0.)

    def test_two_admitted_frames_overlap_without_parallel_state_updates(self):
        rows = [record(0,0.,(.1,.3),(.3,.8),.9),
                record(1,.2,(.3,.6),(.85,1.3),1.4),
                record(2,.91,(.95,1.2),(1.4,1.8),1.9)]
        result = analyze(rows,2)
        self.assertEqual(result['observed_in_flight_peak'],2)
        self.assertAlmostEqual(result['overlap_seconds'],.55)
        self.assertAlmostEqual(result['network_service_seconds'],.75)
        self.assertAlmostEqual(result['post_service_seconds'],1.35)
        with self.assertRaises(ValueError): analyze(rows,1)

    def test_invalid_order_duplicate_frames_and_parallel_stages_rejected(self):
        valid = [record(0,0.,(.1,.3),(.4,.8),.9),record(1,.2,(.3,.6),(.85,1.3),1.4)]
        cases=[]
        rows=copy.deepcopy(valid);rows[1]['frame_index']=0;cases.append(rows)
        rows=copy.deepcopy(valid);rows[1]['wall_intervals']['network']=(.2,.6);cases.append(rows)
        rows=copy.deepcopy(valid);rows[1]['wall_intervals']['post']=(.7,1.3);cases.append(rows)
        rows=copy.deepcopy(valid);rows[0]['wall_intervals']['released']=.7;cases.append(rows)
        rows=copy.deepcopy(valid);rows[0]['wall_intervals']['admitted']=float('nan');cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError): analyze(rows,2)

    def test_empty_run_has_no_leaked_admission(self):
        result = analyze([],2)
        self.assertEqual(result['frames'],0);self.assertEqual(result['observed_in_flight_peak'],0)
        with self.assertRaises(ValueError): analyze([],0)


if __name__=='__main__':unittest.main()
