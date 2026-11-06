import unittest
from .simulation import simulate,summarize


def services(n,p,count):
    return [dict(network=n,post=p,release=p) for _ in range(count)]


class SimulationTests(unittest.TestCase):
    def test_fifo_serial_known_accumulation(self):
        rows,c=simulate(services(.02,.02,4),30,'fifo_serial')
        self.assertEqual([r['frame_index'] for r in rows],[0,1,2,3])
        self.assertAlmostEqual(rows[-1]['result'],.16)
        self.assertAlmostEqual(rows[-1]['result']-rows[-1]['captured'],.06)

    def test_pipeline_all_frames_can_meet_known_deadline(self):
        rows,c=simulate(services(.02,.02,10),30,'fifo_pipeline2')
        result=summarize(rows,c,10,30,100)
        self.assertEqual(result['deadline_misses'],0)
        self.assertAlmostEqual(result['latency_ms']['maximum'],40.)
        self.assertEqual(c['input_skipped'],0)

    def test_latest_serial_skips_capture_but_preserves_source_time(self):
        rows,c=simulate(services(.03,.06,10),30,'latest_serial')
        self.assertEqual([r['frame_index'] for r in rows],[0,2,5,8,9])
        self.assertEqual(c['input_skipped'],5)
        self.assertAlmostEqual(rows[2]['captured'],5/30)

    def test_ready_replacement_and_two_credit_bound(self):
        s=services(.01,.06,20)
        rows,c=simulate(s,30,'latest_ready')
        self.assertGreater(c['ready_replaced'],0)
        self.assertEqual(c['input_skipped'],0)
        self.assertLess(max(r['post_start']-r['network_end'] for r in rows),1/30+1e-8)
        for policy in ('latest_pipeline2','fifo_pipeline2'):
            rows,_=simulate(s,30,policy)
            events=sorted([(r['network_start'],1) for r in rows]+[(r['released'],-1) for r in rows])
            active=0
            for _,change in events:
                active+=change;self.assertLessEqual(active,2);self.assertGreaterEqual(active,0)

    def test_presentation_deadline_and_input_denominators(self):
        rows,c=simulate(services(.02,.02,5),30,'fifo_pipeline2')
        result=summarize(rows,c,5,30,50,10,30)
        self.assertEqual(result['deadline_misses'],5)
        self.assertEqual(result['timely_fraction_of_input'],0)
        self.assertGreater(result['absent_result_ticks'],0)
        self.assertAlmostEqual(result['latency_ms']['mean'],200/3)

    def test_invalid_and_empty_inputs(self):
        self.assertEqual(simulate([],30,'fifo_serial')[0],[])
        for fps in (0,float('nan')):
            with self.assertRaises(ValueError):simulate([],fps,'fifo_serial')
        with self.assertRaises(ValueError):simulate(services(.01,.02,1),30,'unknown')
        with self.assertRaises(ValueError):simulate([dict(network=.01,post=.02,release=.01)],30,'fifo_serial')


if __name__=='__main__':unittest.main()
