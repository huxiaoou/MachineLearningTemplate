if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import random
    from sklearn.model_selection import train_test_split
    from ml import generate_samples, CML, CMLLm, CMLDt, CMLRidgeCV, CMLLgbm

    n = 500
    b: np.ndarray = np.random.randint(low=-3, high=3, size=20)
    v = 10
    random_state = None

    X, y = generate_samples(n=n, b=b, v=v, random_state=random_state)
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.20, shuffle=False)

    m0 = CMLLm(fit_intercept=False)
    m1 = CMLRidgeCV(alphas=(100, 1000), fit_intercept=False)
    m2 = CMLDt(max_depth=2)
    m3 = CMLLgbm(
        boosting_type="gbdt",
        num_leaves=2,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
    )
    models: list[CML] = [m0, m1, m2, m3]
    for m in models:
        m.main(X_trn, X_tst, y_trn, y_tst)

    X1 = pd.DataFrame(data=X)
    X1["cls"] = random.choices(list("abcd"), k=n)
    X1["cls"] = X1["cls"].astype("category")
    y1 = y + np.array([{"a": 100, "b": 200, "c": 300, "d": 400}[z] for z in X1["cls"]])
    X_trn, X_tst, y_trn, y_tst = train_test_split(X1, y1, test_size=0.20, shuffle=False)
    m3 = CMLLgbm(
        boosting_type="gbdt",
        num_leaves=2,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
    )
    m3.main(X_trn, X_tst, y_trn, y_tst)
    yh = m3.core_model.predict(X_tst)

    X_tst["cls"] = "a"
    X_tst["cls"] = X_tst["cls"].astype("category")
    yh = m3.core_model.predict(X_tst)
    print(pd.DataFrame({"y": y_tst, "yh": yh}))
    print("[INF] test a")
    print("-" * 60)

    X_tst["cls"] = ["a"] * 99 + ["e"]
    X_tst["cls"] = X_tst["cls"].astype("category")
    yh = m3.core_model.predict(X_tst)
    print(pd.DataFrame({"y": y_tst, "yh": yh}))
    print("[INF] test last e")
    print("-" * 60)

    X_tst["cls"] = "e"
    X_tst["cls"] = X_tst["cls"].astype("category")
    yh = m3.core_model.predict(X_tst)
    print(pd.DataFrame({"y": y_tst, "yh": yh}))
    print("[INF] test all e")
    print("-" * 60)
