import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tools.fiber_local_dark_evidence import LocalDarkEvidence,features,nearest,color_evidence,public

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/fiber_local_dark_evidence_20261031.json').read_text(encoding='utf-8'))
SEED=json.loads((ROOT/CFG['proposal']['frozen_seed_config']).read_text(encoding='utf-8'))


def scene(gap=False,white=False,y=200):
    f=np.full((400,400,3),70,np.uint8);p=np.zeros((3,400,400),np.float32);p[0]=.99;p[1]=.005;p[2]=.005
    f[y-20:y+21,:231]=[200,90,25];f[y-20:y+21,151:231]=[80,36,10]
    p[0,y-20:y+21,:131]=.005;p[2,y-20:y+21,:131]=.99
    m=np.zeros((400,400),np.uint8);m[p.argmax(0)==2]=2
    if gap:f[y-20:y+21,171:181]=70
    if white:f[y-20:y+21,190:231]=230;p[0,y-20:y+21,190:231]=.795;p[2,y-20:y+21,190:231]=.2
    proposal={'label':1,'origin':np.array([70.,float(y)]),'u':np.array([1.,0.]),'normal':np.array([0.,1.]),
        'ends':np.array([-60.,60.]),'width':40.,'oriented':True,'fov_end_distances':[10.,130.],'_support_ids':list(range(20))}
    return f,m,p,proposal


class LocalDarkTests(unittest.TestCase):
    def observe(self,f,m,p,proposals,arm='local_current_evidence'):
        ob=LocalDarkEvidence(CFG,SEED)
        with patch.object(ob.seeds,'proposals',return_value=(proposals,{'axes_proposed':len(proposals)})):
            return ob.observe(f,m,p,arm)

    def test_low_neural_dark_extension_has_current_color_and_edges(self):
        f,m,p,q=scene();r=self.observe(f,m,p,[q]);right=r['candidates'][0]['profiles'][1]
        xy=np.array(right['primary_xy']);good=np.array(right['supported_probes']);prob=np.array(right['primary_originalP2'])
        self.assertTrue(((xy[...,0]>151)&good&(prob<.1)).any());self.assertIsNone(right['asserted_tip'])

    def test_primary_coordinates_identical_to_geometry_control(self):
        f,m,p,q=scene();a=self.observe(f,m,p,[q],'geometry_window');b=self.observe(f,m,p,[q])
        for x,y in zip(a['candidates'][0]['profiles'],b['candidates'][0]['profiles']):
            np.testing.assert_array_equal(x['primary_xy'],y['primary_xy'])
            self.assertFalse(np.array(x['supported_probes']).any())

    def test_internal_transition_with_continued_edges_not_terminal_tip(self):
        f,m,p,q=scene();r=self.observe(f,m,p,[q]);x=r['candidates'][0]['profiles'][1]
        self.assertTrue(any(x['internal_transition_candidate']));self.assertIsNone(x['asserted_tip'])
        self.assertGreater(sum(x['station_supported']),2)

    def test_disconnected_visible_section_not_gap_filled(self):
        f,m,p,q=scene(gap=True);before=m.copy();r=self.observe(f,m,p,[q]);x=r['candidates'][0]['profiles'][1]
        centers=np.array(x['centers_xy'])[:,0];supported=np.array(x['station_supported'])
        self.assertFalse(supported[(centers>=174)&(centers<=178)].any());self.assertTrue(supported[centers>185].any())
        np.testing.assert_array_equal(m,before);self.assertFalse(r['apply_to_mask'])

    def test_white_region_can_use_neural_evidence_without_blue_veto(self):
        f,m,p,q=scene(white=True);r=self.observe(f,m,p,[q]);x=r['candidates'][0]['profiles'][1]
        self.assertTrue((np.array(x['supported_probes'])&(np.array(x['primary_xy'])[...,0]>195)).any())

    def test_extreme_color_unavailable_not_background_label(self):
        x=np.array([[[0,0,0],[255,255,255],[80,36,10]]],np.float32)
        c=color_evidence(x,np.array([.1,.2,.7]),np.array([1/3]*3),1.)
        self.assertEqual(c['rho_pixel'].tolist(),[[0.,0.,1.]])

    def test_no_anchor_reverts_to_no_candidates_no_mask_edits(self):
        f,m,p,q=scene();p[2]=.0001;r=self.observe(f,m,p,[q])
        self.assertFalse(r['candidates'][0]['anchor']['qualified']);self.assertEqual(r['candidates'][0]['profiles'],[])

    def test_no_seeds_raw_and_empty_observer(self):
        f,m,p,q=scene();r=self.observe(f,m,p,[]);self.assertEqual(r['hypothesis'],'none')
        r=LocalDarkEvidence(CFG,SEED).observe(f,m,p,'raw2');self.assertEqual(r['candidates'],[])

    def test_black_fov_not_fabricated_edge(self):
        f,m,p,q=scene();f[:]=0;r=self.observe(f,m,p,[q]);self.assertFalse(r['candidates'][0]['anchor']['qualified'])

    def test_nearest_bounds_and_native_label_mapping(self):
        a=np.arange(12).reshape(3,4);v,good=nearest(a,[[.49,.49],[.5,.5],[-.01,1],[4,2]])
        self.assertEqual(v[:2].tolist(),[0,5]);self.assertEqual(good.tolist(),[True,True,False,False])

    def test_two_ends_and_fixed_budget(self):
        f,m,p,q=scene();r=self.observe(f,m,p,[q]);profiles=r['candidates'][0]['profiles']
        self.assertEqual([x['end_sign'] for x in profiles],[-1,1])
        self.assertTrue(all(len(x['primary_xy'])<=48 for x in profiles));self.assertLess(profiles[0]['centers_xy'][-1][0],profiles[0]['centers_xy'][0][0])

    def test_stone_conflict_preserved(self):
        f,m,p,q=scene();m[180:221,151:231]=1;before=m.copy();r=self.observe(f,m,p,[q])
        self.assertTrue(any(np.array(x['protected_conflict']).any() for x in r['candidates'][0]['profiles']));np.testing.assert_array_equal(m,before)

    def test_independent_axes_ambiguous_not_forced_selected(self):
        f,m,p,q=scene();f2,m2,p2,q2=scene(y=80);f[60:101]=f2[60:101];m[60:101]=m2[60:101];p[:,60:101]=p2[:,60:101]
        q2['label']=2;q2['_support_ids']=list(range(100,120));r=self.observe(f,m,p,[q,q2])
        self.assertTrue(r['ambiguous']);self.assertEqual(len(r['qualified_axis_groups']),2)

    def test_duplicates_are_metadata_only(self):
        f,m,p,q=scene();q2=copy.deepcopy(q);q2['label']=2;r=self.observe(f,m,p,[q,q2])
        self.assertFalse(r['ambiguous']);self.assertEqual(len(r['candidates']),2);self.assertEqual(len(r['qualified_axis_groups']),1)

    def test_stateless_no_mutation_and_json(self):
        f,m,p,q=scene();before=[x.copy() for x in [f,m,p]];a=self.observe(f,m,p,[q]);self.observe(f*0,m,p,[]);b=self.observe(f,m,p,[q])
        for key in ['candidates','hypothesis']:self.assertEqual(a[key],b[key])
        for x,y in zip([f,m,p],before):np.testing.assert_array_equal(x,y)
        json.dumps(a,allow_nan=False)


if __name__=='__main__':unittest.main()
