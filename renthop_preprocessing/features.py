#!/usr/bin/env python

import re
import pandas as pd
import numpy as np

FEATURES_MAP = {'elevator': 'elevator',
                'cats allowed': r'(?<!\w)cats?(?!\w)|(?<!\w)(?<!no )pets?(?!\w)',
                'dogs allowed': r'(?<!\w)dogs?(?!\w)|(?<!\w)(?<!no )pets?(?!\w)(?!: cats only)',
                'hardwood floors': 'hardwood',
                'doorman': r'(?<!virtual )doorman',
                'dishwasher': 'dishwasher|dw(?!\w)',
                'laundry': r'laundry(?! is on the blo)',
                'no fee': 'no fee',
                'fitness center': r'fitness(?! goals)|gym',
                'pre war': r'pre\s?war',
                'roof deck': 'roof',
                'outdoor space': 'outdoor|garden|patio',
                'dining room': 'dining',
                'high speed internet': r'high.*internet',
                'balcony': r'balcon(y|ies)|private.*terrace',
                'terrace': 'terrace',
                'swimming pool': r'pool(?! table)',
                'new construction': 'new construction',
                'exclusive': r'exclusive( rental)?$',
                'loft': r'(?<!sleep )loft(?! bed)',
                'wheelchair access': 'wheelchair',
                'simplex': 'simplex',
                'fireplace': ['fireplace(?! storage)', 'deco'], # looks for first regex, excluding matches of the second regex
                'lowrise': r'low\s?rise',
                'garage': r'garage|indoor parking',
                'reduced fee': r'(reduced|low) fee',
                'furnished': ['(?<!un)furni', 'deck|inquire|terrace'],
                'multi level': r'multi\s?level|duplex',
                'high ceilings': r'(hig?h|tall) .*ceiling',
                'super': r'(live|site).*super',
                'parking': r'(?<!street )(?<!side )parking(?! available nearby)',
                'renovated': 'renovated',
                'green building': 'green building',
                'storage': 'storage',
                'stainless steel appliances': r'stainless.*(appliance|refrigerator)',
                'concierge': 'concierge',
                'light': r'(?<!\w)(sun)?light(?!\w)',
                'exposed brick': 'exposed brick',
                'eat in kitchen': r'eat.*kitchen',
                'granite kitchen': 'granite kitchen',
                'bike room': r'(?<!citi)(?<!citi )bike',
                'walk in closet': r'walk.*closet',
                'marble bath': r'marble.*bath',
                'valet': 'valet',
                'subway': r'subway|trains?(?!\w)',
                'lounge': 'lounge',
                'short term allowed': 'short term',
                'children\'s playroom': r'(child|kid).*room',
                'no pets': 'no pets',
                'central a/c': r'central a|ac central',
                'luxury building': 'luxur',
                'view': r'(?<!\w)views?(?!\w)|skyline',
                'virtual doorman': 'virtual d',
                'courtyard': 'courtyard',
                'microwave': 'microwave|mw',
                'sauna': 'sauna'}

def _subparser(x):
    x = x.lower().replace('-', ' ').strip()
    if x[0] == '{':
        return [y.replace('"', '').strip() for y in re.findall(r'(?<=\d\s=\s)([^;]+);', x)]
    x = x.split(u'\u2022')
    return [z for y in x for z in re.split(r'[\.\s!;]!*\s+|\s+-\s+|\s*\*\s*', y)]

def _parser(x):
    return [z for z in [y.strip() for y in _subparser(x)] if len(z) > 0]

def _extract_features(features, feature_parser = lambda x: [x.lower()]):
	return [feature for ft in features for feature in feature_parser(ft)]

def _search_regex(regexes):
    if isinstance(regexes, basestring):
        filter_fun = lambda x: re.search(regexes, x) is not None
    else:
        filter_fun = lambda x: re.search(regexes[0], x) is not None and re.search(regexes[1], x) is None
    return lambda x: 1.0 if np.any([filter_fun(ft) for ft in x]) else 0.0

def get_dummies_from_features(series, dtype = np.float32):
    series = series.apply(lambda x: _extract_features(x, _parser))
    dummies = np.zeros((len(series), len(FEATURES_MAP)), dtype = dtype)
    for i, key in enumerate(FEATURES_MAP):
        dummies[:, i] = series.apply(_search_regex(FEATURES_MAP[key]))
    return dummies
