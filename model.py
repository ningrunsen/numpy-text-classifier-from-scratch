"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
def clean_text(text: str) -> str:
    # TODO: Lowercase text and replace non-alphabetic chars with spaces
    text = str.lower(text) 
    text = [ i if str.isalpha(i) else ' ' for i in text]
    return ''.join(text).strip()

# Step 2 - tokenize
def tokenize(text: str) -> list:
    # TODO: Split cleaned text on whitespace into non-empty word tokens
    return text.split()

# Step 3 - tokenize_corpus
def tokenize_corpus(texts: list) -> list:
    # TODO: Apply clean_text and tokenize to every document so the full corpus becomes a list of token lists.
    res =[]
    for text in texts:
        text = clean_text(text)
        text = tokenize(text)
        res.append(text)
    
    return res

# Step 4 - split_train_val_test_indices
def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    # TODO: Produce shuffled index arrays that partition n_samples into train/val/test
    rng = np.random.default_rng(seed)

    idxs = rng.permutation(n_samples)
    n_val = int(n_samples*val_fraction)
    n_test = int(n_samples*test_fraction)

    n_train = n_samples-n_val-n_test

    return (idxs[:n_train],idxs[n_train:n_train+n_val],idxs[-n_test:])

# Step 5 - count_word_frequencies
def count_word_frequencies(tokenized_docs: list) -> dict:
    # TODO: Return a dict mapping each unique token to its total count...
    if not tokenized_docs:
        return {}
    from collections import Counter

    return Counter(x for row in tokenized_docs for x in row)

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    # TODO: Keep the top max_size most frequent words; map each to an index in [0, V).
    n = min(max_size,len(word_counts))
   
    res={}
    sorted_words =sorted(word_counts.items(),key=lambda x: (-x[1],x[0]))
    for i, (word, _) in enumerate(sorted_words[:n]):
        res[word]=i
      
    
    return res

# Step 7 - tokens_to_bow
import numpy as np
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    # TODO: Convert one document's token list into a bag-of-words count vector...
    import numpy as np
    res = np.zeros(len(vocab))

    from collections import Counter
    counts = Counter(tokens)
    
    for token, occ in counts.items():
        if token not in vocab:
            continue
        res[vocab[token]]=occ
    return res

# Step 8 - corpus_to_bow_matrix
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # TODO: Stack per-document BoW vectors into a 2-D count matrix for a whole corpus.
    if not tokenized_docs:
        return np.zeros((0,len(vocab)))
    res =[]
    for doc in tokenized_docs:
        res.append(tokens_to_bow(doc,vocab))
    return np.array(res)

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # TODO: Count docs where each term appears at least once (df, shape (V,))
    
    return np.sum(bow_matrix>0,axis=0)

# Step 10 - compute_idf
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # TODO: Compute smoothed IDF idf_j = log((n_docs + 1) / (df_j + 1)) + 1
    
    return np.log((n_docs+1)/(df+1))+1

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # TODO: Multiply BoW counts by the fitted IDF vector to produce TF-IDF features.
    
    return bow_matrix*idf

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    # TODO: Fit IDF on the training BoW matrix by chaining DF and IDF.
    # compute_document_frequencies(bow_train)

    bow_matrix = compute_document_frequencies(bow_train)
    return compute_idf(bow_matrix,bow_train.shape[0])

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    # TODO: Map logits to probabilities with a numerically stable logistic sigmoid.
    
    return 1/(1+np.exp(-z))

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    
    z = X @ w + b

    return sigmoid(z)

# Step 15 - binary_cross_entropy
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    # TODO: Compute mean binary cross-entropy plus L2 penalty on the weights.
    y_prob = np.clip(y_prob,a_min=1e-15,a_max=1-1e-15)
    bce = -np.mean(y_true*np.log(y_prob) + (1-y_true)*np.log(1-y_prob))
    loss = bce +l2_lambda/2*np.linalg.norm(w)**2
    return loss

# Step 16 - logistic_gradients
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """
    # TODO: Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.
    n = X.shape[0]
    error = y_prob - y_true
    dw = X.T @ error / n + l2_lambda*w
    db = np.mean(error)

    return dw,db

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    # TODO: Return a zero weight vector of shape (n_features,) and bias 0.0
    
    w = np.zeros(n_features)
    b = 0.0
    return (w,b)

# Step 18 - gradient_descent_step
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    # TODO: Run one full-batch gradient descent update; return (w_new, b_new, loss).
    
    y_prob = logistic_predict_proba(X,w,b)
    loss = binary_cross_entropy(y,y_prob,w,l2_lambda)
    dw,db = logistic_gradients(X,y,y_prob,w,l2_lambda)

    new_w = w-lr*dw
    new_b = b -lr*db

    return (new_w,new_b,loss)

# Step 19 - train_logistic_regression
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    # TODO: Initialize params and run n_epochs of full-batch GD, recording loss...
    
    w,b = initialize_logistic_params(X.shape[1])
    losses =[]

    for _ in range(n_epochs):
        w,b,loss = gradient_descent_step(X,y,w,b,lr,l2_lambda)
        losses.append(loss)
    
    return (w,b,losses)

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    # TODO: Convert probabilities to hard binary labels via the threshold...
    
    return (proba>=threshold).astype(int)

# Step 21 - confusion_counts
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    # TODO: Return the four confusion-matrix counts (tp, fp, tn, fn) as Python ints
    

    tp = int(np.sum((y_true == 1)&(y_pred==1)))
    tn = int(np.sum((y_true == 0)&(y_pred==0)))
    fp = int(np.sum((y_true == 0)&(y_pred==1)))
    fn = int(np.sum((y_true == 1)&(y_pred==0)))

    return (tp,fp,tn,fn)

# Step 22 - metrics_from_counts
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    # TODO: Derive precision, recall, F1, and accuracy from confusion counts...
    # import numpy as np
    if fp+tp ==0:
        precision =0.0
    else:
        precision = tp/(fp+tp)
    if tp+fn==0:
        recall =0.0
    else:
        recall = tp/(tp+fn)
    if precision+recall==0:
        f1=0
    else:
        f1= 2*precision*recall/(precision+recall)
    if tp+fp+tn+fn==0:
        accuracy=0.0
    else:
        accuracy = (tp+tn)/(tp+fp+tn+fn)

    return {
        'precision':precision,
        'recall':recall,
        'f1':f1,
        'accuracy':accuracy
    }

# Step 23 - tune_decision_threshold
def tune_decision_threshold(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray = None) -> tuple:
    # TODO: Find the decision threshold that maximizes F1 on validation data.
    

    best_threshold = None
    best_f1 = -1
    if  thresholds is None:
        thresholds = np.linspace(0,1,101)
    for thresh in thresholds:
        y_pred = predict_labels(proba,thresh)
        tp,fp,tn,fn = confusion_counts(y_true,y_pred)
        metrics = metrics_from_counts(tp,fp,tn,fn)
        f1 = metrics['f1']
        if f1>best_f1:
            best_f1 = f1
            best_threshold=thresh
    
    return (best_threshold,best_f1)

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

