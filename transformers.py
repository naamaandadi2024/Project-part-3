import ast
from sklearn.base import BaseEstimator, TransformerMixin

class ActorExperienceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.actor_dict = None

    def fit(self, X, y=None):
        def safe_eval(x):
            if isinstance(x, list): return x
            try: return ast.literal_eval(x)
            except: return []

        actors_series = X['lead_actors_ids'].apply(safe_eval)
        actor_counts = actors_series.explode().value_counts()
        self.actor_dict = actor_counts.to_dict()
        return self

    def transform(self, X):
        X = X.copy()

        def get_cast_stats(actor_list_raw):
            if isinstance(actor_list_raw, list):
                actor_list = actor_list_raw
            elif isinstance(actor_list_raw, str):
                try:
                    actor_list = ast.literal_eval(actor_list_raw)
                except:
                    return np.nan, np.nan, np.nan
            else:
                return np.nan, np.nan, np.nan

            if not actor_list:
                return np.nan, np.nan, np.nan

            counts = [self.actor_dict.get(str(a).strip(), 0) for a in actor_list]
            
            return sum(counts) / len(counts), max(counts), sum(counts)

        stats = X['lead_actors_ids'].apply(get_cast_stats)
        X['avg_actor_experience'] = [s[0] for s in stats]
        X['max_actor_experience'] = [s[1] for s in stats]
        X['sum_actor_experience'] = [s[2] for s in stats]

        return X[['avg_actor_experience', 'max_actor_experience', 'sum_actor_experience']]

    def get_feature_names_out(self, input_features=None):
        return ['avg_actor_experience', 'max_actor_experience', 'sum_actor_experience']