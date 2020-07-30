IE Based Graph Construction
===========================

Test Input
----------
```text
Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye Allen. Allen attended Lakeside 
School, a private school in Seattle, where he befriended Bill Gates, two years younger, with whom he shared an enthusiasm for computers. 
Paul and Bill used a teletype terminal at their high school, Lakeside, to develop their programming skills on several time-sharing computer systems.
```

How to run
----------
```bash
python -m graph4nlp.pytorch.test.graph_construction.test_ie_graph_construction
```

Results
-------
merge_strategy="global"
```text
{'edge_tokens': ['was', 'born', 'to'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['Kenneth', 'Sam', 'Allen'], 'id': 1}}
{'edge_tokens': ['was', 'born', 'on'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['January', '21', '1953'], 'id': 2}}
{'edge_tokens': ['was', 'born', 'in'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['Seattle', 'Washington'], 'id': 3}}
{'edge_tokens': ['school', 'in'], 'src': {'tokens': ['Lakeside', 'School'], 'id': 4}, 'tgt': {'tokens': ['Seattle'], 'id': 5}}
{'edge_tokens': ['befriended'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['Bill', 'Gates'], 'id': 6}}
{'edge_tokens': ['attended'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['private', 'school', 'in', 'Seattle'], 'id': 7}}
{'edge_tokens': ['shared', 'enthusiasm', 'for'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['computers'], 'id': 8}}
{'edge_tokens': ['is', 'in'], 'src': {'tokens': ['private', 'school'], 'id': 9}, 'tgt': {'tokens': ['Seattle'], 'id': 5}}
{'edge_tokens': ['develop', 'Paul', 'programming', 'skills', 'on'], 'src': {'tokens': ['Bill'], 'id': 10}, 'tgt': {'tokens': ['several', 'time'], 'id': 11}}
{'edge_tokens': ['used'], 'src': {'tokens': ['Bill'], 'id': 10}, 'tgt': {'tokens': ['teletype', 'terminal', 'at', 'Paul'], 'id': 12}}
{'edge_tokens': ['global'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['GLOBAL_NODE'], 'id': 13}}
{'edge_tokens': ['global'], 'src': {'tokens': ['Lakeside', 'School'], 'id': 4}, 'tgt': {'tokens': ['GLOBAL_NODE'], 'id': 13}}
{'edge_tokens': ['global'], 'src': {'tokens': ['private', 'school'], 'id': 9}, 'tgt': {'tokens': ['GLOBAL_NODE'], 'id': 13}}
{'edge_tokens': ['global'], 'src': {'tokens': ['Bill'], 'id': 10}, 'tgt': {'tokens': ['GLOBAL_NODE'], 'id': 13}}
is_connected=True
{0: {'node_attr': None, 'token': ['Paul', 'Allen']}, 1: {'node_attr': None, 'token': ['Kenneth', 'Sam', 'Allen']}, 2: {'node_attr': None, 'token': ['January', '21', '1953']}, 3: {'node_attr': None, 'token': ['Seattle', 'Washington']}, 4: {'node_attr': None, 'token': ['Lakeside', 'School']}, 5: {'node_attr': None, 'token': ['Seattle']}, 6: {'node_attr': None, 'token': ['Bill', 'Gates']}, 7: {'node_attr': None, 'token': ['private', 'school', 'in', 'Seattle']}, 8: {'node_attr': None, 'token': ['computers']}, 9: {'node_attr': None, 'token': ['private', 'school']}, 10: {'node_attr': None, 'token': ['Bill']}, 11: {'node_attr': None, 'token': ['several', 'time']}, 12: {'node_attr': None, 'token': ['teletype', 'terminal', 'at', 'Paul']}, 13: {'node_attr': None, 'token': ['GLOBAL_NODE']}}
{'Edges': [(0, 1), (0, 2), (0, 3), (4, 5), (0, 6), (0, 7), (0, 8), (9, 5), (10, 11), (10, 12), (0, 13), (4, 13), (9, 13), (10, 13)]}
```

merge_strategy=None
```text
{'edge_tokens': ['was', 'born', 'to'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['Kenneth', 'Sam', 'Allen'], 'id': 1}}
{'edge_tokens': ['was', 'born', 'on'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['January', '21', '1953'], 'id': 2}}
{'edge_tokens': ['was', 'born', 'in'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['Seattle', 'Washington'], 'id': 3}}
{'edge_tokens': ['school', 'in'], 'src': {'tokens': ['Lakeside', 'School'], 'id': 4}, 'tgt': {'tokens': ['Seattle'], 'id': 5}}
{'edge_tokens': ['befriended'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['Bill', 'Gates'], 'id': 6}}
{'edge_tokens': ['attended'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['private', 'school', 'in', 'Seattle'], 'id': 7}}
{'edge_tokens': ['shared', 'enthusiasm', 'for'], 'src': {'tokens': ['Paul', 'Allen'], 'id': 0}, 'tgt': {'tokens': ['computers'], 'id': 8}}
{'edge_tokens': ['is', 'in'], 'src': {'tokens': ['private', 'school'], 'id': 9}, 'tgt': {'tokens': ['Seattle'], 'id': 5}}
{'edge_tokens': ['develop', 'Paul', 'programming', 'skills', 'on'], 'src': {'tokens': ['Bill'], 'id': 10}, 'tgt': {'tokens': ['several', 'time'], 'id': 11}}
{'edge_tokens': ['used'], 'src': {'tokens': ['Bill'], 'id': 10}, 'tgt': {'tokens': ['teletype', 'terminal', 'at', 'Paul'], 'id': 12}}
is_connected=False
{0: {'node_attr': None, 'token': ['Paul', 'Allen']}, 1: {'node_attr': None, 'token': ['Kenneth', 'Sam', 'Allen']}, 2: {'node_attr': None, 'token': ['January', '21', '1953']}, 3: {'node_attr': None, 'token': ['Seattle', 'Washington']}, 4: {'node_attr': None, 'token': ['Lakeside', 'School']}, 5: {'node_attr': None, 'token': ['Seattle']}, 6: {'node_attr': None, 'token': ['Bill', 'Gates']}, 7: {'node_attr': None, 'token': ['private', 'school', 'in', 'Seattle']}, 8: {'node_attr': None, 'token': ['computers']}, 9: {'node_attr': None, 'token': ['private', 'school']}, 10: {'node_attr': None, 'token': ['Bill']}, 11: {'node_attr': None, 'token': ['several', 'time']}, 12: {'node_attr': None, 'token': ['teletype', 'terminal', 'at', 'Paul']}}
{'Edges': [(0, 1), (0, 2), (0, 3), (4, 5), (0, 6), (0, 7), (0, 8), (9, 5), (10, 11), (10, 12)]}
```
