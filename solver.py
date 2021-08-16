"""
Implementa la clase Solver del los BP.
"""

from functools import cached_property
from itertools import product

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron, Lasso


def equals(y_true, y_pred):
    return all(y_true == y_pred)


def custom_split():
    for test_index in product(range(6), range(6, 12)):
        train_index = tuple(x for x in range(12) if x not in test_index)
        yield train_index, test_index


equals_scorer = make_scorer(equals)
split = list(custom_split())


class BPSolver:
    def __init__(self, bp, n_select=1, alpha_lasso=0, n_lasso=0, first=5,
                 scale=True, **cluster_params):
        self.bp = bp
        self.segment_vect = bp.segment_rep_table
        self.bp_class = bp.problem_class
        self.solutions = {}
        self.cluster_params = cluster_params
        self.y = np.arange(12) < 6
        self.n_select = n_select
        self.alpha_lasso = alpha_lasso
        self.n_lasso = n_lasso
        self.first = first
        self.scale = scale
        self.atts = []
        self.columns = []

    @property
    def raw_X(self):
        if self.bp_class == 0:
            return self.segment_vect
        else:
            return self.bp.cluster_rep(**self.cluster_params)

    @cached_property
    def processed_X(self):
        X = self.raw_X.copy()
        if self.bp_class != 0 and self.first > 0:
            X.drop(X.columns[self.first:], axis=1, inplace=True)
        if self.scale:
            X = pd.DataFrame(RobustScaler().fit_transform(X),
                             columns=X.columns)
        if self.n_select > 0:
            reducer = SelectKBest(f_classif, k=self.n_select).fit(X, self.y)
            self.atts = X.columns[reducer.get_support(True)]
            X = pd.DataFrame(reducer.transform(X), columns=self.atts)
        elif self.alpha_lasso > 0:
            lasso = Lasso(alpha=self.alpha_lasso).fit(X, self.y)
            n_best = np.argpartition(
                abs(lasso.coef_), -self.n_lasso)[-self.n_lasso:]
            self.atts = [X.columns[i]
                         for i in n_best if abs(lasso.coef_[i]) > 0]
            X.drop([col for col in X.columns if col not in self.atts],
                   axis=1, inplace=True)
        self.columns = X.columns
        return X.replace(np.nan, 0)

    def solve(self, clf, n_jobs=6):
        score = cross_val_score(clf, self.processed_X, self.y, cv=split,
                                n_jobs=n_jobs, scoring=equals_scorer)
        self.solutions[clf.__repr__()] = score.mean()

    def default_solve(self):
        default_clfs = (Perceptron(tol=1e-6, random_state=0),
                        SVC(kernel='rbf', gamma='auto', tol=1e-6),
                        GaussianNB(),
                        DecisionTreeClassifier(random_state=1),
                        DummyClassifier(strategy="uniform"))
        for clf in default_clfs:
            self.solve(clf)

    def solved_pd(self):
        return pd.DataFrame(self.solutions, ["Score"]).T


if __name__ == "__main__":
    # Test de velocidad y correctitud en los 100 BP
    import warnings
    from bongard_problem import BongardProblem
    warnings.simplefilter("ignore")

    default_clfs = (Perceptron(tol=1e-6, random_state=0),
                    SVC(kernel='rbf', gamma='auto', tol=1e-6),
                    GaussianNB(),
                    DecisionTreeClassifier(random_state=1),
                    DummyClassifier(strategy="uniform"))
    full_dic = {clf.__repr__(): [] for clf in default_clfs}
    full_dic["ID"] = []

    for i in range(1, 101):
        try:
            BP = BongardProblem(f"bp/#BP{i:03}.jpg")
            solver = BPSolver(BP, n_select=1)
            solver.default_solve()
            for key, val in solver.solutions.items():
                full_dic[key].append(val)
            full_dic["ID"].append(f"{i:03}")
        except Exception as e:
            print(f"Error {e} en {i:03}")
    df = pd.DataFrame(full_dic, index=full_dic["ID"]).drop(["ID"], axis=1)
    print(df)
