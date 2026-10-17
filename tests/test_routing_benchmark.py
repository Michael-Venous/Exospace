"""Check benchmark integrity and independent optimization oracles offline."""
from functools import lru_cache
from html.parser import HTMLParser
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import re
import tempfile
import unittest

from bench.experiments.routing_benchmark import correct, freeze, route
from bench.experiments.routing_cases import make_cases


class Table(HTMLParser):
    def __init__(self, html):
        super().__init__(); self.rows=[]; self.cell=False; self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag=='tr': self.rows.append([])
        if tag in {'td','th'}: self.cell=True; self.rows[-1].append('')

    def handle_endtag(self, tag):
        if tag in {'td','th'}: self.cell=False

    def handle_data(self, data):
        if self.cell: self.rows[-1][-1]+=data


class RoutingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cases=make_cases()

    def test_split_has_no_shared_families_or_ids(self):
        train={c.family for c in self.cases if c.split=='calibration'}
        test={c.family for c in self.cases if c.split=='heldout'}
        self.assertFalse(train & test)
        self.assertEqual(len({c.id for c in self.cases}),36)
        self.assertEqual([c.record() for c in self.cases],[c.record() for c in make_cases()])

    def test_assignment_reference_matches_independent_dynamic_program(self):
        for case in self.cases:
            if case.family!='worker_assignment': continue
            matrix=[list(map(int,row[1:])) for row in Table(case.html).rows[1:]]
            @lru_cache(None)
            def solve(mask):
                i=mask.bit_count()
                if i==len(matrix): return 0,()
                candidates=[]
                for j in range(len(matrix)):
                    if mask>>j&1: continue
                    cost,tail=solve(mask|1<<j)
                    candidates.append((matrix[i][j]+cost,("ABCDEFG"[j],)+tail))
                return min(candidates)
            self.assertEqual(case.answer,list(solve(0)[1]))

    def test_ambiguity_requires_abstention_without_invented_answer(self):
        case=next(c for c in self.cases if c.decision=='ask_user').record()
        self.assertTrue(correct(case,{'decision':'ask_user','answer':[]}))
        self.assertFalse(correct(case,{'decision':'answer','answer':[]}))
        self.assertFalse(correct(case,{'decision':'ask_user','answer':['Personal']}))

    def test_portfolio_reference_matches_independent_dynamic_program(self):
        for case in self.cases:
            if case.family!='project_selection': continue
            items=[(row[0],*map(int,row[1:])) for row in Table(case.html).rows[1:]]
            budget,hours=map(int,re.search(r'Budget at most (\d+); hours at most (\d+)',case.html).groups())
            @lru_cache(None)
            def solve(i,b,h):
                if i==len(items): return 0,()
                label,cost,duration,value=items[i]
                candidates=[solve(i+1,b,h)]
                if cost<=b and duration<=h:
                    tail_value,tail=solve(i+1,b-cost,h-duration)
                    candidates.append((value+tail_value,(label,)+tail))
                return min(candidates,key=lambda x:(-x[0],x[1]))
            self.assertEqual(case.answer,list(solve(0,budget,hours)[1]))

    def test_calendar_reference_matches_datetime_arithmetic(self):
        for case in self.cases:
            if case.family!='midnight_calendar': continue
            values=dict(Table(case.html).rows[1:])
            start=datetime.strptime(values['Original start'],'%H:%M')
            end=start+timedelta(minutes=int(values['Delay in minutes'])+int(values['Duration in minutes']))
            self.assertEqual(case.answer,[end.strftime('%H:%M'),str((end.date()-start.date()).days)])

    def test_cart_reference_matches_decimal_arithmetic(self):
        for case in self.cases:
            if case.family!='cart_total': continue
            rows=Table(case.html).rows[1:]
            subtotal=sum(Decimal(q)*Decimal(p) for _,q,p in rows)
            discount,shipping=re.search(r'Discount (\d+)%. Shipping (\d+\.\d+)',case.html).groups()
            total=(subtotal*(1-Decimal(discount)/100)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)+Decimal(shipping)
            self.assertEqual(case.answer,[f'{total:.2f}'])

    def test_failed_or_uncertain_judgment_cannot_route_to_lite(self):
        policy={'kind':'threshold','lite_below':1.5,'argon_at':2.5,'min_confidence':.7}
        for change in ({'error':'Timeout'},{'ambiguous':True},{'confidence':.69}):
            observation={'error':'','ambiguous':False,'confidence':1.,'score':0.}
            observation.update(change)
            self.assertEqual(route(policy,observation),'flash')

    def test_cannot_fit_after_heldout_results_exist(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)
            (path/'models.jsonl').write_text(json.dumps({'split':'heldout'})+'\n')
            with self.assertRaisesRegex(ValueError,'Heldout outcomes'):
                freeze(path,{})


if __name__=='__main__': unittest.main()
