"""Count exact recovery separately from whether the true token was queried."""
from runtime import LAYERS

METHODS = ('random', 'language_probability', 'kv_surrogate', 'shuffled_embedding_control')
BUDGETS = (200, 220, 240, 300, 400, 600)


def summarize(trials):
    result = {}
    for layer in LAYERS:
        result[str(layer)] = {}
        for scenario in ('uniform_target', 'model_probability_target'):
            cases = [r for r in trials if r['layer']==layer and r['scenario']==scenario]
            by_method = {}
            for method in METHODS:
                by_method[method] = {}
                for budget in BUDGETS:
                    queried_hits, recovered, mismatch, minimum_distances = 0, 0, 0, []
                    for r in cases:
                        t = r['methods'][method]
                        hit = r['target_index'] in t['indices'][:budget]
                        best = min(range(budget), key=lambda i:t['normalized_mse'][i])
                        exact = t['indices'][best]==r['target_index']
                        queried_hits += hit
                        recovered += exact
                        mismatch += hit and not exact
                        minimum_distances.append(t['normalized_mse'][best])
                    by_method[method][str(budget)] = {'trials':len(cases),
                        'hits':queried_hits,'hit_rate':queried_hits/len(cases),
                        'exact_recoveries':recovered,'exact_recovery_rate':recovered/len(cases),
                        'target_queried_but_nearest_kv_wrong':mismatch,
                        'mean_best_normalized_mse':sum(minimum_distances)/len(cases)}
            result[str(layer)][scenario] = by_method
    return result
