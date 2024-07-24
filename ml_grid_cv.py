import numpy as np
import pandas as pd
import scipy.stats as sps
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


def generate_samples(n: int, b: np.ndarray, v: float, random_state: int | None) -> tuple[np.ndarray, np.ndarray]:
    p = len(b)
    x: np.ndarray = sps.norm.rvs(size=(n, p), random_state=random_state)  # type:ignore
    e = sps.norm.rvs(loc=0, scale=v, size=(n,), random_state=random_state)
    y = x @ b + e
    return x, y


class CML:
    def __init__(self, param_grid: dict | list[dict], cv: int = 10) -> None:
        self.estimator = NotImplemented
        self.param_grid = param_grid
        self.cv = cv

    def display(self, r2_trn: float, r2_tst: float):
        print(f"[INF] {self.__str__():<20s}R2 train = {r2_trn:>6.3f} test = {r2_tst:>6.3f}")

    def main(self, x_trn: np.ndarray, x_tst: np.ndarray, y_trn: np.ndarray, y_tst: np.ndarray):
        grid_cv_seeker = GridSearchCV(self.estimator, self.param_grid, cv=self.cv)
        self.estimator = grid_cv_seeker.fit(x_trn, y_trn)
        cv_results = pd.DataFrame(grid_cv_seeker.cv_results_)
        print(cv_results)

        print("-" * 120)
        print(f"[INF] Best estimator with best score = {grid_cv_seeker.best_score_}")
        print(f"[INF] Best estimator params = {grid_cv_seeker.best_params_}")

        # r2_trn_ = grid_cv_seeker.score(x_trn, y_trn)
        # r2_tst_ = grid_cv_seeker.score(x_tst, y_tst)
        # print("-" * 120)
        # self.display(r2_trn_, r2_tst_)  # type:ignore

        r2_trn = self.estimator.score(x_trn, y_trn)
        r2_tst = self.estimator.score(x_tst, y_tst)
        print("-" * 120)
        self.display(r2_trn, r2_tst)  # type:ignore

        print("-" * 120)
        return 0


class CMLRidge(CML):
    def __init__(self, alpha: list[float]) -> None:
        param_grid = {"alpha": alpha}
        super().__init__(param_grid=param_grid)
        self.estimator = Ridge(fit_intercept=False)

    def __str__(self):
        return "Ridge"


class CMLDt(CML):
    def __init__(self, max_depth: list[int]) -> None:
        param_grid = {"max_depth": max_depth}
        super().__init__(param_grid=param_grid)
        self.estimator = DecisionTreeRegressor()

    def __str__(self):
        return "DecisionTree"


class CMLLgbm(CML):
    def __init__(
        self,
        boosting_type: list[str],
        num_leaves: list[int],
        max_depth: list[int],
        learning_rate: list[float],
        n_estimators: list[int],
    ) -> None:
        param_grid = {
            "boosting_type": boosting_type,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
        }
        super().__init__(param_grid=param_grid)
        self.estimator = lgb.LGBMRegressor(verbose=0)


class CMLXgb(CML):
    def __init__(
        self,
        booster: list[str],  # ["gbtree", "gblinear", "dart"]
        n_estimators: list[int],
        max_depth: list[int],
        max_leaves: list[int],
        grow_policy: list[str],  # ["depthwise", "lossguide"]
        learning_rate: list[float],
        objective: list[str],  # ["reg:squarederror", ]
    ) -> None:
        param_grid = {
            "booster": booster,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_leaves": max_leaves,
            "learning_rate": learning_rate,
            "grow_policy": grow_policy,
            "objective": objective,
        }
        super().__init__(param_grid=param_grid)
        self.estimator = xgb.XGBRegressor(verbosity=0)

    def __str__(self):
        return "XGB"
