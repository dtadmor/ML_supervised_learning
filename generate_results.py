import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

graphs_path = './graphs'
if not os.path.exists(graphs_path):
    os.makedirs(graphs_path)

def calc_confusion_vals(y, y_pred):
    tp = (y_pred[y == 1] == 1).sum()
    tn = (y_pred[y == 0] == 0).sum()
    fp = (y_pred[y == 0] == 1).sum()
    fn = (y_pred[y == 1] == 0).sum()
    return tp, tn, fp, fn

def calc_metric(tp, tn, fp, fn, name):
    if name == 'accuracy':
        metric = (tp + tn) / sum([tp, tn, fp, fn])
    elif name == 'recall':
        metric = tp / (tp + fn)
    elif name == 'precision':
        metric = tp / (tp + fp)
    elif name == 'specificity':
        metric = tn / (tn + fp)
    elif name == 'npv':
        metric = tn / (tn + fn)
    elif name == 'f1':
        metric = tp / (tp + (.5 * (fp + fn)))
    return metric

def get_best_val(results, metric):
    mean_results = results.groupby('val')[metric].mean()
    best_val = mean_results.idxmax()
    return best_val

def tune_params(model, X, y, params, metric):

    outcomes = ['tp', 'tn', 'fp', 'fn', metric]
    cols = ['order', 'param', 'val', 'trial', 'train_time', 'pred_train_time', 'pred_val_time']
    cols += [f'train_{o}' for o in outcomes]
    cols += [f'val_{o}' for o in outcomes]
    results = pd.DataFrame(columns=cols, dtype=int)
    best_params = {}
    if model.__name__ not in {'KNeighborsClassifier'}:
        best_params = {'random_state': 42}
    if model.__name__ == 'MLPClassifier':
        best_params['max_iter'] = 1000
    kfolds = StratifiedKFold(5, shuffle=True, random_state=42)
    i = -1
    for param, vals in params.items():
        i += 1
        param_results = {col: [] for col in cols}
        print(param)
        for val in vals:
            print(val)
            model_inst = model(**{param: val}, **best_params)
            for j, (train_idx, val_idx) in enumerate(kfolds.split(X, y)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                param_results['order'].append(i)
                if model.__name__ == 'AdaBoostClassifier' and param == 'base_estimator':
                    param_results['param'].append('max_depth')
                    param_results['val'].append(val.max_depth)
                else:
                    param_results['param'].append(param)
                    param_results['val'].append(val)
                param_results['trial'].append(j)
                t1 = time.time()
                model_inst.fit(X_train, y_train)
                t2 = time.time()
                param_results['train_time'].append(t2 - t1)
                for X_inst, y_inst, pred_type in zip([X_train, X_val], [y_train, y_val], ['train', 'val']):
                    t1 = time.time()
                    y_pred = model_inst.predict(X_inst)
                    t2 = time.time()
                    param_results[f'pred_{pred_type}_time'].append(t2 - t1)
                    tp, tn, fp, fn = calc_confusion_vals(y_inst, y_pred)
                    param_results[f'{pred_type}_tp'].append(tp)
                    param_results[f'{pred_type}_tn'].append(tn)
                    param_results[f'{pred_type}_fp'].append(fp)
                    param_results[f'{pred_type}_fn'].append(fn)
        for pred_type in ['train', 'val']:
            param_results[f'{pred_type}_{metric}'] = np.nan
        param_results = pd.DataFrame(param_results)
        for pred_type in ['train', 'val']:
            param_results[f'{pred_type}_{metric}'] = calc_metric(
                param_results[f'{pred_type}_tp'],
                param_results[f'{pred_type}_tn'],
                param_results[f'{pred_type}_fp'],
                param_results[f'{pred_type}_fn'],
                metric
            )
        best_val = get_best_val(param_results, f'val_{metric}')
        if model.__name__ == 'AdaBoostClassifier' and param == 'base_estimator':
            best_params[param] = DecisionTreeClassifier(max_depth=best_val)
        else:
            best_params[param] = best_val
        results = pd.concat([results, param_results], axis=0)
    results.reset_index(drop=True, inplace=True)
    return results, best_params

def graph_param_tuning(results, metric, data_name):

    orders = results['order'].unique()
    fig, ax = plt.subplots(1, len(orders), figsize=(16, 5), sharey=True)
    fig.suptitle(f'{data_name} Parameter Tuning', fontsize=16)
    ax[0].set_ylabel(metric.capitalize())
    for i in sorted(orders):
        param_results = results.query('order == @i')
        param = param_results['param'].iloc[0]
        for j, (label, pred_type),  in enumerate(zip(['Train', 'Validation'], ['train', 'val'])):
            scores = calc_metric(
                param_results[f'{pred_type}_tp'],
                param_results[f'{pred_type}_tn'],
                param_results[f'{pred_type}_fp'],
                param_results[f'{pred_type}_fn'],
                metric
            )
            mean_scores = scores.groupby(param_results['val']).mean()
            if pd.api.types.is_numeric_dtype(mean_scores.index.dtype):
                ax[i].plot(mean_scores.index, mean_scores.values, marker='o', label=label)
            else:
                w = .35
                x = np.arange(len(mean_scores.index)) + ((j - .5) * w)
                ax[i].bar(x, mean_scores.values, width=w, label=label)
                ax[i].set_xticks(x - ((j - .5) * w))
                ax[i].set_xticklabels(mean_scores.index)
        param_name = param.replace('_', ' ').title()
        ax[i].set_title(f'{param_name} Tuning')
        ax[i].set_xlabel(param_name)
        ax[i].legend()
    fname = f'./graphs/{data_name} Tuning'
    plt.savefig(fname)

def calc_performance_over_training_size(model, X, y, params):

    pcts = np.linspace(.2, 1, 5)
    outcomes = ['tp', 'tn', 'fp', 'fn']
    cols = ['trial', 'percent', 'train_time', 'pred_train_time', 'pred_val_time']
    cols += [f'train_{o}' for o in outcomes]
    cols += [f'val_{o}' for o in outcomes]
    results = pd.DataFrame(columns=cols, dtype=int)
    model_inst = model(**params)
    kfolds = StratifiedKFold(5, shuffle=True, random_state=42)
    for pct in pcts:
        if pct < 1:
            X_sub, _, y_sub, _ = train_test_split(X, y, stratify=y, train_size=pct, shuffle=True, random_state=42)
        elif pct == 1:
            X_sub = X.copy()
            y_sub = y.copy()
        param_results = {col: [] for col in cols}
        for i, (train_idx, val_idx) in enumerate(kfolds.split(X_sub, y_sub)):
            param_results['percent'].append(pct)
            X_train = X_sub[train_idx]
            y_train = y_sub[train_idx]
            X_val = X_sub[val_idx]
            y_val = y_sub[val_idx]
            param_results['trial'].append(i)
            t1 = time.time()
            model_inst.fit(X_train, y_train)
            t2 = time.time()
            param_results['train_time'].append(t2 - t1)
            for X_inst, y_inst, pred_type in zip([X_train, X_val], [y_train, y_val], ['train', 'val']):
                t1 = time.time()
                y_pred = model_inst.predict(X_inst)
                t2 = time.time()
                param_results[f'pred_{pred_type}_time'].append(t2 - t1)
                tp, tn, fp, fn = calc_confusion_vals(y_inst, y_pred)
                param_results[f'{pred_type}_tp'].append(tp)
                param_results[f'{pred_type}_tn'].append(tn)
                param_results[f'{pred_type}_fp'].append(fp)
                param_results[f'{pred_type}_fn'].append(fn)
        param_results = pd.DataFrame(param_results)
        results = pd.concat([results, param_results], axis=0)
    results.reset_index(drop=True, inplace=True)
    return results

def graph_performance_over_training_size(results, metrics, data_name):

    fig, ax = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=True)
    fig.suptitle(f'{data_name} Learning Curve Over Training Size', fontsize=16)
    ax[0].set_ylabel('Score')
    for i, metric in enumerate(metrics):
        for label, pred_type in zip(['Train', 'Validation'], ['train', 'val']):
            scores = calc_metric(
                results[f'{pred_type}_tp'],
                results[f'{pred_type}_tn'],
                results[f'{pred_type}_fp'],
                results[f'{pred_type}_fn'],
                metric
            )
            mean_scores = scores.groupby(results['percent']).mean()
            ax[i].plot(mean_scores.index, mean_scores.values, marker='o', label=label)
        ax[i].set_title(metric.title())
        ax[i].set_xlabel('Training Size')
        ax[i].set_xticklabels([f'{int(p*100)}%' for p in ax[i].get_xticks()])
        ax[i].legend()
    fname = f'./graphs/{data_name} Sizes'
    plt.savefig(fname)

def calc_learning_curve(model, X, y, params, svm_step=2000):

    iters = np.linspace(.2, 1, 5)
    outcomes = ['tp', 'tn', 'fp', 'fn']
    cols = ['iteration']
    cols += [f'train_{o}' for o in outcomes]
    cols += [f'val_{o}' for o in outcomes]
    results = pd.DataFrame(columns=cols, dtype=int)
    model_inst = model(**params)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, train_size=.8, shuffle=True, random_state=42)

    if model.__name__ == 'AdaBoostClassifier':
        model_inst.fit(X_train, y_train)
        train_preds = model_inst.staged_predict(X_train)
        val_preds = model_inst.staged_predict(X_val)
    elif model.__name__ == 'MLPClassifier':
        classes = np.unique(y_train)
        train_preds = []
        val_preds = []
        for i in range(params['max_iter']):
            model_inst.partial_fit(X_train, y_train, classes)
            train_preds.append(model_inst.predict(X_train))
            val_preds.append(model_inst.predict(X_val))
    elif model.__name__ == 'SVC':
        train_preds = []
        val_preds = []
        for i in range(svm_step, svm_step*20 + 1, svm_step):
            model_inst.set_params(max_iter=i)
            model_inst.fit(X_train, y_train)
            train_preds.append(model_inst.predict(X_train))
            val_preds.append(model_inst.predict(X_val))

    for i, (y_train_pred, y_val_pred) in enumerate(zip(train_preds, val_preds), 1):
        iter_results = {}
        if model.__name__ == 'SVC':
            iter_results['iteration'] = [i * svm_step]
        else:
            iter_results['iteration'] = [i]
        for y_true, y_pred, pred_type in zip([y_train, y_val], [y_train_pred, y_val_pred], ['train', 'val']):
            tp, tn, fp, fn = calc_confusion_vals(y_true, y_pred)
            iter_results[f'{pred_type}_tp'] = [tp]
            iter_results[f'{pred_type}_tn'] = [tn]
            iter_results[f'{pred_type}_fp'] = [fp]
            iter_results[f'{pred_type}_fn'] = [fn]
        iter_results = pd.DataFrame(iter_results)
        results = pd.concat([results, iter_results], axis=0)

    results.reset_index(drop=True, inplace=True)
    return results

def graph_learning_curve(results, metrics, data_name):

    fig, ax = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=True)
    fig.suptitle(f'{data_name} Learning Curve Over Training Iterations', fontsize=16)
    ax[0].set_ylabel('Score')
    for i, metric in enumerate(metrics):
        for label, pred_type in zip(['Train', 'Validation'], ['train', 'val']):
            scores = calc_metric(
                results[f'{pred_type}_tp'],
                results[f'{pred_type}_tn'],
                results[f'{pred_type}_fp'],
                results[f'{pred_type}_fn'],
                metric
            )
            ax[i].plot(results['iteration'], scores.values, marker='o', label=label)
        ax[i].set_title(metric.title())
        ax[i].set_xlabel('Iteration')
        ax[i].legend()
    fname = f'./graphs/{data_name} Iterations'
    plt.savefig(fname)

with open('./census/adult.names') as f:
    names = f.readlines()
cols = [c for c in names if c[0] != '|']
cols = [c.replace('\n', '') for c in cols]
cols = [c.split(':')[0] for c in cols]
cols = [c for c in cols if c]
cols = cols[1:] + [cols[0]]
df_census = pd.read_csv('./census/adult.data', names=cols)

X_cols = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]
categorical_cols = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'
]
X_census = pd.get_dummies(df_census[X_cols], columns=categorical_cols, drop_first=True)
y_census = df_census['>50K, <=50K.'].map(lambda x: {'>50K': 1, '<=50K': 0}.get(x.strip()))
ss = StandardScaler()
x_cols = X_census.columns
X_census[x_cols] = ss.fit_transform(X_census)

df_spam = pd.read_csv('./spam/spambase.data', header=None)

X_cols = df_spam.columns[:-1]
y_spam = df_spam[57]
ss = StandardScaler()
X_spam = pd.DataFrame(ss.fit_transform(df_spam[X_cols]), columns=X_cols)

X_census_train, X_census_test, y_census_train, y_census_test = train_test_split(
    X_census, y_census, stratify=y_census, train_size=.8, shuffle=True, random_state=42
)
X_spam_train, X_spam_test, y_spam_train, y_spam_test = train_test_split(
    X_spam, y_spam, stratify=y_spam, train_size=.8, shuffle=True, random_state=42
)

results_spam = {
    'dt': {},
    'knn': {},
    'ada': {},
    'svm': {},
    'nn': {}
}
results_census = {
    'dt': {},
    'knn': {},
    'ada': {},
    'svm': {},
    'nn': {}
}

model = DecisionTreeClassifier
params_census = {
    'ccp_alpha': [0, .00005, .0001, .001],
    'max_depth': [5, 10, 20, 40],
    'criterion': ['gini', 'entropy'],
}
params_spam = {
    'max_depth': [5, 10, 20, 40],
    'ccp_alpha': [0, .00005, .0001, .001],
    'criterion': ['gini', 'entropy'],
}
tune_results, best_params = tune_params(model, X_census_train.values, y_census_train.values, params_census, 'accuracy')
graph_param_tuning(tune_results, 'accuracy', 'Census Decision Tree')
size_results = calc_performance_over_training_size(model, X_census_train.values, y_census_train.values, best_params)
metrics = ['accuracy', 'recall', 'precision']
graph_performance_over_training_size(size_results, metrics, 'Census Decision Tree')
results_census['dt'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results
}
tune_results, best_params = tune_params(model, X_spam_train.values, y_spam_train.values, params_spam, 'accuracy')
graph_param_tuning(tune_results, 'accuracy', 'Spam Decision Tree')
params_spam = {
    'ccp_alpha': [0, .00005, .0001, .001],
    'max_depth': [5, 10, 20, 40],
    'criterion': ['gini', 'entropy'],
}
size_results = calc_performance_over_training_size(model, X_spam_train.values, y_spam_train.values, best_params)
metrics = ['accuracy', 'recall', 'precision']
graph_performance_over_training_size(size_results, metrics, 'Spam Decision Tree')
results_spam['dt'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results
}

model = KNeighborsClassifier
params_census = {
    'n_neighbors': [1, 3, 5],
    'weights': ['uniform', 'distance']
}
params_spam = {
    'n_neighbors': [1, 3, 5],
    'weights': ['uniform', 'distance']
}

tune_results, best_params = tune_params(
    model, X_census_train.values, y_census_train.values, params_census, 'accuracy'
)

graph_param_tuning(tune_results, 'accuracy', 'Census KNN')

size_results = calc_performance_over_training_size(model, X_census_train.values, y_census_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
scores = graph_performance_over_training_size(size_results, metrics, 'Census KNN')

results_census['knn'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results
}

tune_results, best_params = tune_params(
    model, X_spam_train.values, y_spam_train.values, params_spam, 'accuracy'
)

graph_param_tuning(tune_results, 'accuracy', 'Spam KNN')

size_results = calc_performance_over_training_size(model, X_spam_train.values, y_spam_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
scores = graph_performance_over_training_size(size_results, metrics, 'Spam KNN')

results_spam['knn'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results
}

model = AdaBoostClassifier
params_census = {
    'learning_rate': [.01, .1, 1],
    'n_estimators': [10, 25, 50, 100],
    'base_estimator': [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2),
        DecisionTreeClassifier(max_depth=5),
        DecisionTreeClassifier(max_depth=10)
    ]
}
params_spam = {
    'learning_rate': [.01, .1, 1],
    'n_estimators': [10, 25, 50, 100],
    'base_estimator': [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2),
        DecisionTreeClassifier(max_depth=5),
        DecisionTreeClassifier(max_depth=10)
    ]
}

tune_results, best_params = tune_params(
    model, X_census_train.values, y_census_train.values, params_census, 'accuracy'
)

graph_param_tuning(tune_results, 'accuracy', 'Census AdaBoost')

size_results = calc_performance_over_training_size(model, X_census_train.values, y_census_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
graph_performance_over_training_size(size_results, metrics, 'Census AdaBoost')

learn_results = calc_learning_curve(model, X_census_train.values, y_census_train.values, best_params)

graph_learning_curve(learn_results, metrics, 'Census AdaBoost')

results_census['ada'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results,
    'iter': learn_results
}

tune_results, best_params = tune_params(
    model, X_spam_train.values, y_spam_train.values, params_spam, 'accuracy'
)

graph_param_tuning(tune_results, 'accuracy', 'Spam AdaBoost')

size_results = calc_performance_over_training_size(model, X_spam_train.values, y_spam_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
graph_performance_over_training_size(size_results, metrics, 'Spam AdaBoost')

learn_results = calc_learning_curve(model, X_spam_train.values, y_spam_train.values, best_params)

graph_learning_curve(learn_results, metrics, 'Spam AdaBoost')

results_spam['ada'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results,
    'iter': learn_results
}

model = SVC
params_census = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'C': [.001, .01, .1, 1],
}
params_spam = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'C': [.001, .01, .1, 1],
}

tune_results, best_params = tune_params(
    model, X_census_train.values, y_census_train.values, params_census, 'accuracy'
)

graph_param_tuning(tune_results, 'accuracy', 'Census SVM')

size_results = calc_performance_over_training_size(model, X_census_train.values, y_census_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
graph_performance_over_training_size(size_results, metrics, 'Census SVM')

learn_results = calc_learning_curve(model, X_census_train.values, y_census_train.values, best_params)

graph_learning_curve(learn_results, metrics, 'Census SVM')

results_census['svm'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results,
    'iter': learn_results
}

tune_results, best_params = tune_params(
    model, X_spam_train.values, y_spam_train.values, params_spam, 'accuracy'
)

graph_param_tuning(tune_results, 'accuracy', 'Spam SVM')

size_results = calc_performance_over_training_size(model, X_spam_train.values, y_spam_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
graph_performance_over_training_size(size_results, metrics, 'Spam SVM')

learn_results = calc_learning_curve(model, X_spam_train.values, y_spam_train.values, best_params, svm_step=50)

graph_learning_curve(learn_results, metrics, 'Spam SVM')

results_spam['svm'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results,
    'iter': learn_results
}

model = MLPClassifier
params_census = {
    'learning_rate_init': [.001, .005, .01],
    'hidden_layer_sizes': [50, 100, 300],
    'activation': ['identity', 'logistic', 'relu']
}
params_spam = {
    'learning_rate_init': [.001, .005, .01],
    'hidden_layer_sizes': [50, 100, 300],
    'activation': ['identity', 'logistic', 'relu']
}

tune_results, best_params = tune_params(
    model, X_census_train.values, y_census_train.values, params_census, 'accuracy'
)

graph_param_tuning(tune_results, 'accuracy', 'Census NN')

size_results = calc_performance_over_training_size(model, X_census_train.values, y_census_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
graph_performance_over_training_size(size_results, metrics, 'Census NN')

learn_results = calc_learning_curve(model, X_census_train.values, y_census_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
graph_learning_curve(learn_results, metrics, 'Census NN')

results_census['nn'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results,
    'iter': learn_results
}

tune_results, best_params = tune_params(
    model, X_spam_train.values, y_spam_train.values, params_spam, 'accuracy'
)

graph_param_tuning(tune_results, 'accuracy', 'Spam NN')

size_results = calc_performance_over_training_size(model, X_spam_train.values, y_spam_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
graph_performance_over_training_size(size_results, metrics, 'Spam NN')

learn_results = calc_learning_curve(model, X_spam_train.values, y_spam_train.values, best_params)

metrics = ['accuracy', 'recall', 'precision']
graph_learning_curve(learn_results, metrics, 'Spam NN')

results_spam['nn'] = {
    'best_params': best_params,
    'tune': tune_results,
    'size': size_results,
    'iter': learn_results
}

model_dict = {
    'dt': DecisionTreeClassifier,
    'knn': KNeighborsClassifier,
    'ada': AdaBoostClassifier,
    'svm': SVC,
    'nn': MLPClassifier
}
results_cols = ['model', 'data', 'train_time', 'pred_time', 'accuracy', 'recall', 'precision']
final_results = pd.DataFrame(index=range(len(model_dict)*2), columns=results_cols)

for i, (name, model) in enumerate(model_dict.items()):

    final_results.loc[i, 'model'] = name
    final_results.loc[i, 'data'] = 'census'
    census_params = results_census[name]['best_params']
    census_model = model(**census_params)
    t1 = time.time()
    census_model.fit(X_census_train, y_census_train)
    t2 = time.time()
    final_results.loc[i, 'train_time'] = t2 - t1
    t1 = time.time()
    y_pred = census_model.predict(X_census_test)
    t2 = time.time()
    final_results.loc[i, 'pred_time'] = t2 - t1
    tp, tn, fp, fn = calc_confusion_vals(y_census_test, y_pred)
    for metric_name in ['accuracy', 'recall', 'precision']:
        metric = calc_metric(tp, tn, fp, fn, metric_name)
        final_results.loc[i, metric_name] = metric

    final_results.loc[i+5, 'model'] = name
    final_results.loc[i+5, 'data'] = 'spam'
    spam_params = results_spam[name]['best_params']
    spam_model = model(**spam_params)
    t1 = time.time()
    spam_model.fit(X_spam_train, y_spam_train)
    t2 = time.time()
    final_results.loc[i+5, 'train_time'] = t2 - t1
    t1 = time.time()
    y_pred = spam_model.predict(X_spam_test)
    t2 = time.time()
    final_results.loc[i+5, 'pred_time'] = t2 - t1
    tp, tn, fp, fn = calc_confusion_vals(y_spam_test, y_pred)
    for metric_name in ['accuracy', 'recall', 'precision']:
        metric = calc_metric(tp, tn, fp, fn, metric_name)
        final_results.loc[i+5, metric_name] = metric

for model_name, model_results in results_census.items():
    for results_name, data in model_results.items():
        path = f'./results/census/{model_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, f'{results_name}.csv')
        if isinstance(data, pd.DataFrame):
            data.to_csv(full_path)
        elif isinstance(data, dict):
            pd.DataFrame(data, index=[0]).to_csv(full_path)

for model_name, model_results in results_spam.items():
    for results_name, data in model_results.items():
        path = f'./results/spam/{model_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, f'{results_name}.csv')
        if isinstance(data, pd.DataFrame):
            data.to_csv(full_path)
        elif isinstance(data, dict):
            pd.DataFrame(data, index=[0]).to_csv(full_path)

path = f'./results/final/'
if not os.path.exists(path):
    os.makedirs(path)
full_path = os.path.join(path, 'testing.csv')
final_results.to_csv(full_path)
