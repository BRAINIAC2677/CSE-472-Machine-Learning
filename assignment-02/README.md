# Ensembling Logistic Regression

## Class Structure and Inheritance

This notebook contains a variety of machine learning models, organized in an object-oriented hierarchy. The classes inherit from each other to promote reusability and extension of common methods.

### Class Diagram

```plaintext
          ┌─────────────────────────────────┐
          │             Model (ABC)         │
          └─────────────────────────────────┘
                          ▲
                          │
          ┌─────────────────────────────────┐
          │        BinaryClassifier         │
          └─────────────────────────────────┘
               ▲              ▲
               │              │
 ┌────────────────────────┐ ┌────────────────────────┐
 │    LogisticRegressor   │ │      Ensembler         │
 └────────────────────────┘ └────────────────────────┘
                                     ▲
                 ┌────────────┬──────┴───────┬───────────┐
                 │            │              │           │
   ┌─────────────────────┐ ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────────┐
   │  MeanEnsembler      │ │  MultipleRegressor │ │  VotingEnsembler   │ │  StackingEnsembler     │
   └─────────────────────┘ └────────────────────┘ └────────────────────┘ └────────────────────────┘


    class Model {
        +np.ndarray train_x
        +np.ndarray train_y
        +dict config
        __init__(train_x: np.ndarray, train_y: np.ndarray, config: dict)
        fit()
        predict(x: np.ndarray)
        test(x: np.ndarray, y: np.ndarray)
    }
    
    class BinaryClassifier {
        +np.ndarray val_x
        +np.ndarray val_y
        __init__(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, config: dict)
        h(x: np.ndarray)
        calculate_metrics(_x: np.ndarray, _y: np.ndarray)
    }
    
    class LogisticRegressor {
        -np.ndarray __w
        __init__(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, config: dict)
        __miniBGD(_lr: float, _epoch: int, _batch_size: int, _verbose: bool, l1_lambda: float, l2_lambda: float)
        __negative_log_likelihood(_x: np.ndarray, _y: np.ndarray)
        h(x: np.ndarray)
        fit()
        predict(x: np.ndarray)
        test(x: np.ndarray, y: np.ndarray)
    }
    
    class Bootstrapper {
        +np.ndarray train_x
        +np.ndarray train_y
        +np.ndarray val_x
        +np.ndarray val_y
        +list estimators
        __init__(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray)
        __bootstrap_sample()
        __fit_estimators(config: dict)
        get_estimators(config: dict)
    }
    
    class Ensembler {
        +Bootstrapper _bootstrapper
        +list _estimators
        __init__(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, config: dict, estimators: list)
        __get_all_metrics()
        draw_violin_plot(figsize: tuple, title: str)
        fit()
        predict(x: np.ndarray)
        test(x: np.ndarray, y: np.ndarray)
    }

    class MeanEnsembler {
        __init__(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, config: dict, estimators: list)
        h(x: np.ndarray)
    }
    
    class MultipleRegressor {
        __init__(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, config: dict, estimators: list)
        calculate_metrics(_x: np.ndarray, _y: np.ndarray)
        h(x: np.ndarray)
    }

    class VotingEnsembler {
        __init__(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, config: dict, estimators: list)
        h(x: np.ndarray)
        predict(x: np.ndarray)
    }

    class StackingEnsembler {
        -np.ndarray __meta_x
        -np.ndarray __meta_y
        -LogisticRegressor __meta_learner
        __init__(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, config: dict, estimators: list)
        __generate_meta_x(x: np.ndarray)
        __generate_metaset()
        __fit_meta_learner(_lr: float, _epoch: int, _batch_size: int, _verbose: bool)
        h(x: np.ndarray)
        fit()
        predict(x: np.ndarray)
    }
    
