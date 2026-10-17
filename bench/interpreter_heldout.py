"""Fresh labels/requests, frozen before inspecting optimized-study outcomes."""
from bench import interpreter_study as study

study.ELEMENTS.update({
    'search':'input: Lookup','address':'input: Destination','notes':'input: Quick memo',
    'open':'button: Launch','save':'button: Keep draft','tools':'menu button: Actions',
    'first':'link: Seaside morning (first result)','second':'link: Thunderstorm (second result)',
    'folder':'folder: Fieldwork','archive':'drop region: Storage'})
a=study.action;c=study.case
study.CASES[:]=[
    c('h1','hey put north star into Quick memo and keep the draft',a('fill','notes','north star'),a('click','save')),
    c('h2','in Lookup, enter cats & dogs',a('fill','search','cats & dogs')),
    c('h3','set Destination to https://example.org/a?x=1&y=2 and hit Launch',a('fill','address','https://example.org/a?x=1&y=2'),a('click','open')),
    c('h4','Quick memo should contain exactly "A, B; C."',a('fill','notes','A, B; C.')),
    c('h5','clear the Quick memo field',a('fill','notes','')),
    c('h6','click Launch, no wait, Keep draft instead',a('click','save')),
    c('h7','please double click Fieldwork then hover over Actions',a('double_click','folder'),a('hover','tools')),
    c('h8','continue with the Work account please',a('click','continue_b')),
    c('h9','continue with my account',status='clarify'),
    c('h10','compose a two sentence thank you in Quick memo',status='planner'),
    c('h11','put a German translation of good morning in Quick memo',status='planner'),
    c('h12','I almost clicked Launch by mistake',status='none'),
]

if __name__=='__main__':
    from bench.interpreter_optimized import main
    main()
