import numpy as np
import scipy.stats as sps
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor


def generate_samples(n: int, b: np.ndarray, v: float, random_state: int | None) -> tuple[np.ndarray, np.ndarray]:
    p = len(b)
    x: np.ndarray = sps.norm.rvs(size=(n, p), random_state=random_state)  # type:ignore
    e = sps.norm.rvs(loc=0, scale=v, size=(n,), random_state=random_state)
    y = x @ b + e
    return x, y


class CML:
    def __init__(self) -> None:
        self.core_model = NotImplemented

    def display(self, r2_trn: float, r2_tst: float):
        print(f"[INF] {self.__str__():<20s}R2 train = {r2_trn:>6.3f} test = {r2_tst:>6.3f}")

    def main(self, x_trn: np.ndarray, x_tst: np.ndarray, y_trn: np.ndarray, y_tst: np.ndarray):
        self.core_model.fit(x_trn, y_trn)
        r2_trn = self.core_model.score(x_trn, y_trn)
        r2_tst = self.core_model.score(x_tst, y_tst)
        self.display(r2_trn, r2_tst)
        return 0


class CMLLm(CML):
    def __init__(self, fit_intercept: bool) -> None:
        super().__init__()
        self.core_model = LinearRegression(fit_intercept=fit_intercept)

    def __str__(self):
        return "LinearModel"


class CMLRidgeCV(CML):
    def __init__(self, alphas: tuple, fit_intercept: bool) -> None:
        super().__init__()
        self.core_model = RidgeCV(alphas=alphas, fit_intercept=fit_intercept)

    def __str__(self):
        return "RidgeCV"


class CMLDt(CML):
    def __init__(self, max_depth: int) -> None:
        super().__init__()
        self.core_model = DecisionTreeRegressor(max_depth=max_depth)

    def __str__(self):
        return "DecisionTree"


class CMLLgbm(CML):
    def __init__(
            self, boosting_type: str, num_leaves: int, max_depth: int, learning_rate: float, n_estimators: int,
    ) -> None:
        super().__init__()
        self.core_model = lgb.LGBMRegressor(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            force_col_wise=True,
            verbose=0,
        )

    def __str__(self):
        return "LGBM"
