if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    from ml_grid_cv import generate_samples, CML, CMLDt, CMLRidge, CMLLgbm, CMLXgb

    n = 500
    b: np.ndarray = np.random.randint(low=-3, high=3, size=20)
    v = 10
    random_state = None

    X, y = generate_samples(n=n, b=b, v=v, random_state=random_state)
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.20, shuffle=False)

    m1 = CMLRidge(alpha=[100, 1000])
    m2 = CMLDt(max_depth=[2, 4])
    m3 = CMLLgbm(
        boosting_type=["gbdt"],
        num_leaves=[2],
        max_depth=[-1],
        learning_rate=[0.01, 0.1],
        n_estimators=[100],
    )
    m4 = CMLXgb(
        booster=["gbtree", "gblinear", "dart"],
        n_estimators=[50, 100],
        max_depth=[2, 4],
        max_leaves=[4, 16],
        grow_policy=["depthwise", "lossguide"],
        learning_rate=[0.01, 0.1],
        objective=["reg:squarederror"],
    )
    models: list[CML] = [m1, m2, m3, m4]
    for m in models:
        print("\n" + "=" * 120)
        m.main(X_trn, X_tst, y_trn, y_tst)
