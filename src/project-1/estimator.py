from numba import jit


@jit
def point632plus(train_score, test_score, r_marked, test_score_marked):

    point632 = 0.368 * train_score + 0.632 * test_score
    frac = (0.368 * 0.632 * r_marked) / (1 - 0.368 * r_marked)

    return point632 + (test_score_marked - train_score) * frac


@jit
def relative_overfit_rate(train_score, test_score, gamma):

    if test_score > train_score and gamma > train_score:
        return (test_score - train_score) / (gamma - train_score)
    
    return 0


def no_info_rate(y_true, y_pred):

    # NB: Need only use sum(y) if y is binary.
    p_one = np.sum(y_true == 1) / np.size(y_true)
    q_one = np.sum(y_pred == 1) / np.size(y_pred)

    return p_one * (1 - q_one) + (1 - p_one) * q_one


def point632plus_score(y_true, y_pred, train_score, test_score):

    gamma = no_info_rate(y_true, y_pred)
    
    # To account for gamma <= train_score/train_score < gamma <= test_score
    # in which case r can fall outside of [0, 1].
    test_score_marked = min(test_score, gamma)

    r_adjusted = relative_overfit_rate(train_score, test_score, gamma)

    # Compute .632+ score.
    return point632plus(train_score, test_score, r_adjusted, test_score_marked)


from sklearn.metrics import matthews_corrcoef


def point_estimate():
    """A single estimate for model performance."""
    pass


def resampling_estimate(score_func=None):
    """Model performance estimate bounded by a confidence interval.
    
    Args:
        score_func: Performance metric. Defaults to Matthews correlation 
            coefficient.
    
    Returns:
        Training (to check for overfitting) and test performance.
    """
    # Use 
    
    data_sampler = BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    
    if score_func is None:
        score_func = matthews_corrcoef
    
    train_scores, test_scores = [], []
    for num, (train_idx, test_idx) in enumerate(data_sampler.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train_std)
        train_score = score_func(y_train, y_train_pred)

        y_test_pred = model.predict(X_test_std)
        test_score = score_func(y_test, y_test_pred)

        train_632_score = point632plus_score(
            y_train, y_train_pred, train_score, test_score
        )
        
        test_632_score = point632plus_score(
            y_test, y_test_pred, train_score, test_score
        )
        
        train_scores.append(train_632_score)
        test_scores.append(test_632_score)
        
    return train_scores, test_scores
