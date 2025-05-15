# Module redxclassifiers.classical_ml_models

Module for classical regression model defaults

??? example "View Source"
        #!/usr/bin.env python3

        # -*- coding: utf-8 -*-

        """

        Module for classical regression model defaults

        """

        import logging

        from dataclasses import dataclass

        from datetime import datetime

        from typing import Any, Callable, List, Optional, Tuple, Union

        import lightgbm

        import numpy as np

        import sklearn

        import xgboost

        from lightgbm import LGBMClassifier

        from sklearn.ensemble import (

            AdaBoostClassifier,

            ExtraTreesClassifier,

            HistGradientBoostingClassifier,

            RandomForestClassifier,

        )

        from sklearn.gaussian_process import GaussianProcessClassifier

        from sklearn.gaussian_process.kernels import RBF, Matern

        from sklearn.linear_model import LogisticRegression

        from sklearn.neural_network import MLPClassifier

        from sklearn.svm import SVC

        from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

        from xgboost import XGBClassifier

        from redxclassifiers import utilities

        log = logging.getLogger(__name__)



        # Data class to contain an sklearn Classifier model. The class allows for a default hyper parameter grid to trained over or for a custom one to be added without loosing the default.

        @dataclass

        class skmodel:

            name: str

            model: Union[Any, Callable]

            default_param_finite_grid: dict

            default_param_range_priors: dict

            custom_param_grid: dict

            multi_output_Classifier: bool

            more_info: Optional[str] = None

            sklearn_version: str = sklearn.__version__

            date: str = datetime.strftime(datetime.now(), "%d/%m/%Y")



        def get_models(

            linear: bool = True,

            kernel: bool = True,

            bayesian: bool = True,

            ensemble: bool = True,

            neural_network: bool = True,

            n_features: Optional[int] = None,

            prepend_parameter_keys_with_model: bool = False,

            multi_output_only: bool = False,

            prepend_with: str = "model__",

            **kwargs,

        ) -> List[skmodel]:

            """

            Function to return a list of models to train over and trial

            Args:

                linear (bool, optional): Return linear models. Defaults to True.

                kernel (bool, optional): Return kernel models. Defaults to True.

                bayesian (bool, optional): Return Bayesian models. Defaults to True.

                ensemble (bool, optional): Return ensemble models. Defaults to True.

                neural_network (bool, optional): Return neural network models. Defaults to True.

                n_features (Optional[int], optional): The number of input features. Used in Bayesian model Gaussian process to allow the use for anisotropic kernels. Defaults to None.

            Returns:

                List: List of sklearn models wrapped in a custom dataclass with meta data

            """

            models = []

            if linear is True:

                models = models + linear_models(**kwargs)

            if kernel is True:

                models = models + kernel_models(**kwargs)

            if bayesian is True:

                models = models + bayesian_models(n_features=n_features, **kwargs)

            if ensemble is True:

                models = models + ensemble_models(**kwargs)

            if neural_network is True:

                models = models + neural_network_models(n_input_features=n_features, **kwargs)

            if prepend_parameter_keys_with_model is True:

                for m in models:

                    m.default_param_grid = {

                        f"{prepend_with}{k}": v for k, v in m.default_param_grid.items()

                    }

                    m.custom_param_grid = {

                        f"{prepend_with}{k}": v for k, v in m.custom_param_grid.items()

                    }

            if multi_output_only is True:

                models = [m for m in models if m.multi_output_Classifier is True]

            return models



        def linear_models(logreg: bool = True, lr: bool = True, **kwargs) -> List[skmodel]:

            """

            Function to build a default set of linear models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of linear models and meta data

            """

            models = []

            if logreg is True or lr is True:

                # Linear regression

                models.append(

                    skmodel(

                        type(LogisticRegression()).__name__,

                        LogisticRegression(n_jobs=-1),

                        {

                            "penalty": ["l1", "l2", "elasticnet", "none"],

                            "C": [0.5, 1.0, 1.5],

                            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],

                            "fit_intercept": [True, False],

                        },

                        {

                            "penalty": ["catagorical", ["l1", "l2", "elasticnet", "none"]],

                            "C": ["float", [0.0, 1.5]],

                            "solver": [

                                "catagorical",

                                ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],

                            ],

                            "fit_intercept": ["catagorical", [True, False]],

                        },

                        {},

                        True,

                        "hhttps://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            return models



        def kernel_models(svc: bool = True, svm: bool = False, **kwargs) -> List[skmodel]:

            """

            Function to build a default set of kernel models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of kernel models and meta data

            """

            models = []

            if svc is True or svm is True:

                # Support Vector Regression (SVR)

                models.append(

                    skmodel(

                        type(SVC()).__name__,

                        SVC(max_iter=10000),

                        {

                            "kernel": ["rbf", "linear"],

                            "C": [0.9, 1.0, 1.1],

                            "epsilon": [0.05, 0.1, 0.2],

                            "gamma": [None, 0.5, 1.5],

                        },

                        {

                            "kernel": ["catagorical", ["rbf", "linear"]],

                            "C": ["float", 0.01, 5.0],

                            "epsilon": ["float", 0.01, 1.0],

                            "gamma": ["float", 0.01, 5.0],

                        },

                        {},

                        False,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

                # Support Vector Regression (SVR) - polynomical kernel has the degree parameter so we separate the model

                models.append(

                    skmodel(

                        type(SVC()).__name__ + "poly_kernel",

                        SVC(max_iter=10000),

                        {

                            "kernel": ["poly"],

                            "C": [0.9, 1.0, 1.1],

                            "epsilon": [0.05, 0.1, 0.2],

                            "degree": [2, 3, 4, 5],

                            "gamma": [None, 0.5, 1.5],

                        },

                        {

                            "kernel": ["catagorical", ["poly"]],

                            "C": ["float", 0.01, 5.0],

                            "epsilon": ["float", 0.01, 1.0],

                            "degree": ["int", 2, 8],

                            "gamma": ["float", 0.01, 5.0],

                        },

                        {},

                        False,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            return models



        def bayesian_models(

            gp: bool = True,

            n_features: Optional[int] = None,

            n_restarts_optimizer: int = 5,

            **kwargs,

        ) -> List[skmodel]:

            """

            Function to build a default set of Bayesian models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of Bayesian models and meta data

            """

            models = []

            if gp is True:

                # Gaussian process

                # If we know the number of features we can use an anisotropic kernel amd an isotropic kernel otherwise we can only use an isotropic kernel

                if n_features is not None:

                    ls = np.array([1.0] * n_features)

                    # NOTE: None uses the default Constant kernel(1.0) * RBF(length_scale=1.0) with fixed length scales. The last kernel allows length scale optimization

                    # using the RBF kernel

                    kernels = [

                        None,

                        Matern(length_scale=ls, nu=1.5),

                        Matern(length_scale=ls, nu=2.5),

                        Matern(length_scale=ls, nu=0.5),

                        RBF(length_scale=ls),

                        Matern(length_scale=1.0, nu=1.5),

                        Matern(length_scale=1.0, nu=2.5),

                        Matern(length_scale=1.0, nu=0.5),

                        RBF(length_scale=1.0),

                    ]

                else:

                    kernels = [

                        None,

                        Matern(length_scale=1.0, nu=1.5),

                        Matern(length_scale=1.0, nu=2.5),

                        Matern(length_scale=1.0, nu=0.5),

                        RBF(length_scale=1.0),

                    ]

                models.append(

                    skmodel(

                        type(GaussianProcessClassifier()).__name__,

                        GaussianProcessClassifier(

                            random_state=utilities.random_seed,

                            n_restarts_optimizer=n_restarts_optimizer,

                        ),

                        {"kernel": kernels},

                        {

                            "kernels": ["catagorical", kernels],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        True,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            return models



        def ensemble_models(

            adb: bool = True,

            rf: bool = True,

            et: bool = True,

            hgb: bool = True,

            xgb: bool = True,

            lgbm: bool = True,

            **kwargs,

        ) -> List[skmodel]:

            """

            Function to build a default set of ensemble models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of ensemble models and meta data

            """

            models = []

            if rf is True:

                # Random forest

                models.append(

                    skmodel(

                        type(RandomForestClassifier()).__name__,

                        RandomForestClassifier(

                            random_state=utilities.random_seed, n_jobs=-1, oob_score=True

                        ),

                        {

                            "n_estimators": [50, 100, 200, 500],

                            "ccp_alpha": [0.0, 0.05],

                            "max_features": [0.1, "sqrt", "log2", 0.3, 0.7, 1.0],

                            "max_depth": [3, 5, 7, 9],

                            "max_samples": [0.3, 0.5, 0.8],

                        },

                        {

                            "n_estimators": ["int", 50, 700],

                            "ccp_alpha": ["float", 0.0, 1.0],

                            "max_features": ["float", 0.1, 1.0],

                            "max_depth": ["int", 3, 15],

                            "max_samples": ["float", 0.1, 1.0],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        True,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            if et is True:

                # Extra trees Classifier

                models.append(

                    skmodel(

                        type(ExtraTreesClassifier()).__name__,

                        ExtraTreesClassifier(random_state=utilities.random_seed, n_jobs=-1),

                        {

                            "n_estimators": [50, 100, 200, 500],

                            "ccp_alpha": [0.0, 0.05],

                            "max_features": [0.1, "sqrt", "log2", 0.3, 0.7, 1.0],

                            "max_depth": [3, 5, 7, 9],

                        },

                        {

                            "n_estimators": ["int", 50, 700],

                            "ccp_alpha": ["float", 0.0, 1.0],

                            "max_features": ["float", 0.1, 1.0],

                            "max_depth": ["int", 3, 15],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        True,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            if adb is True:

                # Adaboost

                models.append(

                    skmodel(

                        type(AdaBoostClassifier()).__name__,

                        AdaBoostClassifier(random_state=utilities.random_seed),

                        {

                            "estimator": [

                                DecisionTreeClassifier(max_depth=3),

                                ExtraTreeClassifier(max_depth=3),

                            ],

                            "n_estimators": [30, 50, 70],

                            "learning_rate": [0.01, 0.1, 0.2, 0.4],

                        },

                        {

                            "n_estimators": ["int", 20, 700],

                            "learning_rate": ["float", 0.001, 0.4],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        False,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            if hgb is True:

                # Gradient boosting - This version is similar to LightGBM

                models.append(

                    skmodel(

                        type(HistGradientBoostingClassifier()).__name__,

                        HistGradientBoostingClassifier(random_state=utilities.random_seed),

                        {

                            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],

                            "max_leaf_nodes": [10, 31, 50],

                            "max_depth": [3, 5, None],

                            "l2_regularization": [0.0, 0.5, 1.0],

                            "max_iter": [50, 100, 150],

                            "loss": ["squared_error", "absolute_error"],

                        },

                        {

                            "learning_rate": ["float", 0.001, 0.4],

                            "max_leaf_nodes": ["int", 10, 100],

                            "max_depth": ["int", 3, 15],

                            "l2_regularization": ["float", 0.0, 1.0],

                            "max_iter": ["int", 50, 200],

                            "loss": ["catagorical", ["squared_error", "absolute_error"]],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        False,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            if xgb is True:

                # EXtreme Gradient Boosting - Model tuning guide https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning

                # early stopping https://xgboosting.com/configure-xgboost-early-stopping-regularization/

                # training https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/#:~:text=The%20number%20of%20trees%20(or,the%20XGBoost%20library%20is%20100.

                models.append(

                    skmodel(

                        type(XGBClassifier(random_state=utilities.random_seed)).__name__,

                        XGBClassifier(

                            random_state=utilities.random_seed, n_jobs=-1, verbosity=0

                        ),

                        {

                            "learning_rate": [0.05, 0.1, 0.2, 0.3],

                            "n_estimators": [100, 200, 300],

                            "max_depth": [3, 5, 7, 10],

                            "min_child_weight": [1, 5, 7],

                            "gamma": [0.1, 0.25, 0.5],

                            "subsample": [0.7, 1.0],

                            "colsample_bytree": [0.8, 1.0],

                            "reg_alpha": [0.05, 1.0, 10.0],

                            "reg_lambda": [0.05, 1.0, 10.0],

                        },

                        {

                            "learning_rate": ["float", 0.001, 0.4],

                            "n_estimators": ["int", 50, 400],

                            "max_depth": ["int", 3, 15],

                            "min_child_weight": ["int", 1, 10],

                            "gamma": ["float", 0.01, 0.5],

                            "subsample": ["float", 0.1, 1.0],

                            "colsample_bytree": ["float", 0.1, 1.0],

                            "reg_alpha": ["float", 0.01, 10.0],

                            "reg_lambda": ["float", 0.05, 10.0],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        False,

                        "https://xgboost.readthedocs.io/en/stable/python/python_api.html",

                        f"XGBoost version: {xgboost.__version__}",

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

                # {

                #             "n_estimators": [100, 200, 500],

                #             "max_depth": [3, 5, 7],

                #             "min_child_weight": [1, 2, 3],

                #             "gamma": [0, 0.05, 0.1],

                #             "subsample": [0.6, 0.8, 1.0],

                #             "colsample_bytree": [0.6, 0.8, 1.0],

                #             "reg_alpha": [0.0, 0.1, 0.2],

                #             "reg_lambda": [1.0, 1.1, 1.2],

                #         },

                # {

                #     "n_estimators": [100, 250, 500, 750, 1000],

                #     "learning_rate": [0.2, 0.3, 0.4],

                #     "max_depth": [3, 5, 7],

                #     "min_child_weight": [1, 2, 3],

                #     "gamma": [0, 0.05, 0.1],

                #     "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                #     "colsample_bytree": [0.6, 0.8, 1.0],

                #     "reg_alpha": [0.0, 0.1, 0.25, 0.4],

                #     "reg_lambda": [1.0, 1.1, 1.25, 1.4],

                #     "grow_policy": ["depthwise", "lossguide"],

                # },

            if lgbm:

                # Light Gradient Boosting Machine - Model tuning guide https://www.kaggle.com/code/bextuychiev/lgbm-optuna-hyperparameter-tuning-w-understanding

                models.append(

                    skmodel(

                        type(LGBMClassifier()).__name__,

                        LGBMClassifier(

                            random_state=utilities.random_seed, n_jobs=-1, verbose=-1

                        ),

                        {

                            "learning_rate": [0.01, 0.05, 0.1, 0.2],

                            "n_estimators": [100, 150, 200],

                            "max_depth": [-1, 3, 5, 7, 9],

                            "num_leaves": [20, 31, 100],

                            "min_data_in_leaf": [10, 20, 50, 70],

                            "reg_alpha": [0.0, 1.0, 10.0],

                            "reg_lambda": [0.0, 1.0, 10.0],

                        },

                        {

                            "learning_rate": ["float", 0.001, 0.4],

                            "n_estimators": ["int", 50, 400],

                            "max_depth": ["int", 3, 15],

                            "num_leaves": ["int", 20, 100],

                            "min_data_in_leaf": ["int", 10, 100],

                            "reg_alpha": ["float", 0.01, 10.0],

                            "reg_lambda": ["float", 0.05, 10.0],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        False,

                        "https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html",

                        f"LightGBM version: {lightgbm.__version__}",

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

                # {

                #             "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],

                #             "n_estimators": [100, 150, 200],

                #             "max_depth": [-1, 3, 5, 7],

                #             "num_leaves": [15, 31, 45, 60],

                #             "min_child_samples": [10, 20, 30],

                #             "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                #             "colsample_bytree": [0.6, 0.8, 1.0],

                #             "reg_alpha": [0.0, 1.0, 10.0, 100.0],

                #             "reg_lambda": [0.0, 1.0, 10.0, 100.0],

                #         },

            return models



        def neural_network_models(

            mlp: bool = True,

            layers: Tuple[Tuple[int, int, int], ...] = (

                (20, 10, 5),

                (50, 50, 50),

                (100, 50, 10),

                (100, 50, 25),

                (100, 50, 40),

            ),

            solver: str = "adam",

            max_iter: int = 400,

            n_input_features: Optional[int] = None,

            skip_feature_n_dependent: bool = False,

            **kwargs,

        ) -> List[skmodel]:

            """

             Function to build a default set of neural network models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of neural network models and meta data

            """

            models = []

            if isinstance(layers, tuple):

                _layers = list(layers)

            if n_input_features is not None and skip_feature_n_dependent is False:

                if isinstance(layers, tuple):

                    _layers = list(layers)

                hl1 = [int(n_input_features * elt) for elt in [0.5, 0.7, 0.9]]

                hl2 = [int(n_input_features * elt) for elt in [0.3, 0.4, 0.5]]

                hl3 = [int(n_input_features * elt) for elt in [0.1, 0.2, 0.4]]

                for ith in range(3):

                    _layers.append((hl1[ith], hl2[ith], hl3[ith]))

            if mlp is True:

                # Multi-layer perceptron (MLP) - This is the only general NN in sklearn currently. It is built as feedforward network.

                models.append(

                    skmodel(

                        type(MLPClassifier()).__name__,

                        MLPClassifier(

                            max_iter=max_iter, solver=solver, random_state=utilities.random_seed

                        ),

                        {

                            "learning_rate": ["constant", "invscaling", "adaptive"],

                            "learning_rate_init": [0.0001, 0.001, 0.01],

                            "alpha": [0.00001, 0.0001, 0.0005, 0.001],

                            "activation": ["tanh", "relu"],

                            "hidden_layer_sizes": _layers,

                            "batch_size": ["auto", 16, 32],

                        },

                        {

                            "learning_rate": [

                                "catagorical",

                                ["constant", "invscaling", "adaptive"],

                            ],

                            "learning_rate_init": ["float", 0.0001, 0.2],

                            "alpha": ["float", 0.00001, 0.5],

                            "activation": [

                                "catagorical",

                                [

                                    "tanh",

                                    "relu",

                                ],

                            ],

                            "n_layers": ["int", 1, 5],

                            "batch_size": ["int", 10, 100],

                        },

                        {},

                        True,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            return models



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### bayesian_models

```python3
def bayesian_models(
    gp: bool = True,
    n_features: Optional[int] = None,
    n_restarts_optimizer: int = 5,
    **kwargs
) -> List[classical_ml_models.skmodel]
```

Function to build a default set of Bayesian models and parameter grids to optimize the models over

**Returns:**

| Type | Description |
|---|---|
| List[skmodel] | List of Bayesian models and meta data |

??? example "View Source"
        def bayesian_models(

            gp: bool = True,

            n_features: Optional[int] = None,

            n_restarts_optimizer: int = 5,

            **kwargs,

        ) -> List[skmodel]:

            """

            Function to build a default set of Bayesian models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of Bayesian models and meta data

            """

            models = []

            if gp is True:

                # Gaussian process

                # If we know the number of features we can use an anisotropic kernel amd an isotropic kernel otherwise we can only use an isotropic kernel

                if n_features is not None:

                    ls = np.array([1.0] * n_features)

                    # NOTE: None uses the default Constant kernel(1.0) * RBF(length_scale=1.0) with fixed length scales. The last kernel allows length scale optimization

                    # using the RBF kernel

                    kernels = [

                        None,

                        Matern(length_scale=ls, nu=1.5),

                        Matern(length_scale=ls, nu=2.5),

                        Matern(length_scale=ls, nu=0.5),

                        RBF(length_scale=ls),

                        Matern(length_scale=1.0, nu=1.5),

                        Matern(length_scale=1.0, nu=2.5),

                        Matern(length_scale=1.0, nu=0.5),

                        RBF(length_scale=1.0),

                    ]

                else:

                    kernels = [

                        None,

                        Matern(length_scale=1.0, nu=1.5),

                        Matern(length_scale=1.0, nu=2.5),

                        Matern(length_scale=1.0, nu=0.5),

                        RBF(length_scale=1.0),

                    ]

                models.append(

                    skmodel(

                        type(GaussianProcessClassifier()).__name__,

                        GaussianProcessClassifier(

                            random_state=utilities.random_seed,

                            n_restarts_optimizer=n_restarts_optimizer,

                        ),

                        {"kernel": kernels},

                        {

                            "kernels": ["catagorical", kernels],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        True,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            return models


### ensemble_models

```python3
def ensemble_models(
    adb: bool = True,
    rf: bool = True,
    et: bool = True,
    hgb: bool = True,
    xgb: bool = True,
    lgbm: bool = True,
    **kwargs
) -> List[classical_ml_models.skmodel]
```

Function to build a default set of ensemble models and parameter grids to optimize the models over

**Returns:**

| Type | Description |
|---|---|
| List[skmodel] | List of ensemble models and meta data |

??? example "View Source"
        def ensemble_models(

            adb: bool = True,

            rf: bool = True,

            et: bool = True,

            hgb: bool = True,

            xgb: bool = True,

            lgbm: bool = True,

            **kwargs,

        ) -> List[skmodel]:

            """

            Function to build a default set of ensemble models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of ensemble models and meta data

            """

            models = []

            if rf is True:

                # Random forest

                models.append(

                    skmodel(

                        type(RandomForestClassifier()).__name__,

                        RandomForestClassifier(

                            random_state=utilities.random_seed, n_jobs=-1, oob_score=True

                        ),

                        {

                            "n_estimators": [50, 100, 200, 500],

                            "ccp_alpha": [0.0, 0.05],

                            "max_features": [0.1, "sqrt", "log2", 0.3, 0.7, 1.0],

                            "max_depth": [3, 5, 7, 9],

                            "max_samples": [0.3, 0.5, 0.8],

                        },

                        {

                            "n_estimators": ["int", 50, 700],

                            "ccp_alpha": ["float", 0.0, 1.0],

                            "max_features": ["float", 0.1, 1.0],

                            "max_depth": ["int", 3, 15],

                            "max_samples": ["float", 0.1, 1.0],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        True,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            if et is True:

                # Extra trees Classifier

                models.append(

                    skmodel(

                        type(ExtraTreesClassifier()).__name__,

                        ExtraTreesClassifier(random_state=utilities.random_seed, n_jobs=-1),

                        {

                            "n_estimators": [50, 100, 200, 500],

                            "ccp_alpha": [0.0, 0.05],

                            "max_features": [0.1, "sqrt", "log2", 0.3, 0.7, 1.0],

                            "max_depth": [3, 5, 7, 9],

                        },

                        {

                            "n_estimators": ["int", 50, 700],

                            "ccp_alpha": ["float", 0.0, 1.0],

                            "max_features": ["float", 0.1, 1.0],

                            "max_depth": ["int", 3, 15],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        True,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            if adb is True:

                # Adaboost

                models.append(

                    skmodel(

                        type(AdaBoostClassifier()).__name__,

                        AdaBoostClassifier(random_state=utilities.random_seed),

                        {

                            "estimator": [

                                DecisionTreeClassifier(max_depth=3),

                                ExtraTreeClassifier(max_depth=3),

                            ],

                            "n_estimators": [30, 50, 70],

                            "learning_rate": [0.01, 0.1, 0.2, 0.4],

                        },

                        {

                            "n_estimators": ["int", 20, 700],

                            "learning_rate": ["float", 0.001, 0.4],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        False,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            if hgb is True:

                # Gradient boosting - This version is similar to LightGBM

                models.append(

                    skmodel(

                        type(HistGradientBoostingClassifier()).__name__,

                        HistGradientBoostingClassifier(random_state=utilities.random_seed),

                        {

                            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],

                            "max_leaf_nodes": [10, 31, 50],

                            "max_depth": [3, 5, None],

                            "l2_regularization": [0.0, 0.5, 1.0],

                            "max_iter": [50, 100, 150],

                            "loss": ["squared_error", "absolute_error"],

                        },

                        {

                            "learning_rate": ["float", 0.001, 0.4],

                            "max_leaf_nodes": ["int", 10, 100],

                            "max_depth": ["int", 3, 15],

                            "l2_regularization": ["float", 0.0, 1.0],

                            "max_iter": ["int", 50, 200],

                            "loss": ["catagorical", ["squared_error", "absolute_error"]],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        False,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            if xgb is True:

                # EXtreme Gradient Boosting - Model tuning guide https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning

                # early stopping https://xgboosting.com/configure-xgboost-early-stopping-regularization/

                # training https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/#:~:text=The%20number%20of%20trees%20(or,the%20XGBoost%20library%20is%20100.

                models.append(

                    skmodel(

                        type(XGBClassifier(random_state=utilities.random_seed)).__name__,

                        XGBClassifier(

                            random_state=utilities.random_seed, n_jobs=-1, verbosity=0

                        ),

                        {

                            "learning_rate": [0.05, 0.1, 0.2, 0.3],

                            "n_estimators": [100, 200, 300],

                            "max_depth": [3, 5, 7, 10],

                            "min_child_weight": [1, 5, 7],

                            "gamma": [0.1, 0.25, 0.5],

                            "subsample": [0.7, 1.0],

                            "colsample_bytree": [0.8, 1.0],

                            "reg_alpha": [0.05, 1.0, 10.0],

                            "reg_lambda": [0.05, 1.0, 10.0],

                        },

                        {

                            "learning_rate": ["float", 0.001, 0.4],

                            "n_estimators": ["int", 50, 400],

                            "max_depth": ["int", 3, 15],

                            "min_child_weight": ["int", 1, 10],

                            "gamma": ["float", 0.01, 0.5],

                            "subsample": ["float", 0.1, 1.0],

                            "colsample_bytree": ["float", 0.1, 1.0],

                            "reg_alpha": ["float", 0.01, 10.0],

                            "reg_lambda": ["float", 0.05, 10.0],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        False,

                        "https://xgboost.readthedocs.io/en/stable/python/python_api.html",

                        f"XGBoost version: {xgboost.__version__}",

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

                # {

                #             "n_estimators": [100, 200, 500],

                #             "max_depth": [3, 5, 7],

                #             "min_child_weight": [1, 2, 3],

                #             "gamma": [0, 0.05, 0.1],

                #             "subsample": [0.6, 0.8, 1.0],

                #             "colsample_bytree": [0.6, 0.8, 1.0],

                #             "reg_alpha": [0.0, 0.1, 0.2],

                #             "reg_lambda": [1.0, 1.1, 1.2],

                #         },

                # {

                #     "n_estimators": [100, 250, 500, 750, 1000],

                #     "learning_rate": [0.2, 0.3, 0.4],

                #     "max_depth": [3, 5, 7],

                #     "min_child_weight": [1, 2, 3],

                #     "gamma": [0, 0.05, 0.1],

                #     "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                #     "colsample_bytree": [0.6, 0.8, 1.0],

                #     "reg_alpha": [0.0, 0.1, 0.25, 0.4],

                #     "reg_lambda": [1.0, 1.1, 1.25, 1.4],

                #     "grow_policy": ["depthwise", "lossguide"],

                # },

            if lgbm:

                # Light Gradient Boosting Machine - Model tuning guide https://www.kaggle.com/code/bextuychiev/lgbm-optuna-hyperparameter-tuning-w-understanding

                models.append(

                    skmodel(

                        type(LGBMClassifier()).__name__,

                        LGBMClassifier(

                            random_state=utilities.random_seed, n_jobs=-1, verbose=-1

                        ),

                        {

                            "learning_rate": [0.01, 0.05, 0.1, 0.2],

                            "n_estimators": [100, 150, 200],

                            "max_depth": [-1, 3, 5, 7, 9],

                            "num_leaves": [20, 31, 100],

                            "min_data_in_leaf": [10, 20, 50, 70],

                            "reg_alpha": [0.0, 1.0, 10.0],

                            "reg_lambda": [0.0, 1.0, 10.0],

                        },

                        {

                            "learning_rate": ["float", 0.001, 0.4],

                            "n_estimators": ["int", 50, 400],

                            "max_depth": ["int", 3, 15],

                            "num_leaves": ["int", 20, 100],

                            "min_data_in_leaf": ["int", 10, 100],

                            "reg_alpha": ["float", 0.01, 10.0],

                            "reg_lambda": ["float", 0.05, 10.0],

                            "random_state": [

                                "int",

                                utilities.random_seed,

                                utilities.random_seed,

                            ],

                        },

                        {},

                        False,

                        "https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html",

                        f"LightGBM version: {lightgbm.__version__}",

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

                # {

                #             "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],

                #             "n_estimators": [100, 150, 200],

                #             "max_depth": [-1, 3, 5, 7],

                #             "num_leaves": [15, 31, 45, 60],

                #             "min_child_samples": [10, 20, 30],

                #             "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                #             "colsample_bytree": [0.6, 0.8, 1.0],

                #             "reg_alpha": [0.0, 1.0, 10.0, 100.0],

                #             "reg_lambda": [0.0, 1.0, 10.0, 100.0],

                #         },

            return models


### get_models

```python3
def get_models(
    linear: bool = True,
    kernel: bool = True,
    bayesian: bool = True,
    ensemble: bool = True,
    neural_network: bool = True,
    n_features: Optional[int] = None,
    prepend_parameter_keys_with_model: bool = False,
    multi_output_only: bool = False,
    prepend_with: str = 'model__',
    **kwargs
) -> List[classical_ml_models.skmodel]
```

Function to return a list of models to train over and trial

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| linear | bool | Return linear models. Defaults to True. | True |
| kernel | bool | Return kernel models. Defaults to True. | True |
| bayesian | bool | Return Bayesian models. Defaults to True. | True |
| ensemble | bool | Return ensemble models. Defaults to True. | True |
| neural_network | bool | Return neural network models. Defaults to True. | True |
| n_features | Optional[int] | The number of input features. Used in Bayesian model Gaussian process to allow the use for anisotropic kernels. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| List | List of sklearn models wrapped in a custom dataclass with meta data |

??? example "View Source"
        def get_models(

            linear: bool = True,

            kernel: bool = True,

            bayesian: bool = True,

            ensemble: bool = True,

            neural_network: bool = True,

            n_features: Optional[int] = None,

            prepend_parameter_keys_with_model: bool = False,

            multi_output_only: bool = False,

            prepend_with: str = "model__",

            **kwargs,

        ) -> List[skmodel]:

            """

            Function to return a list of models to train over and trial

            Args:

                linear (bool, optional): Return linear models. Defaults to True.

                kernel (bool, optional): Return kernel models. Defaults to True.

                bayesian (bool, optional): Return Bayesian models. Defaults to True.

                ensemble (bool, optional): Return ensemble models. Defaults to True.

                neural_network (bool, optional): Return neural network models. Defaults to True.

                n_features (Optional[int], optional): The number of input features. Used in Bayesian model Gaussian process to allow the use for anisotropic kernels. Defaults to None.

            Returns:

                List: List of sklearn models wrapped in a custom dataclass with meta data

            """

            models = []

            if linear is True:

                models = models + linear_models(**kwargs)

            if kernel is True:

                models = models + kernel_models(**kwargs)

            if bayesian is True:

                models = models + bayesian_models(n_features=n_features, **kwargs)

            if ensemble is True:

                models = models + ensemble_models(**kwargs)

            if neural_network is True:

                models = models + neural_network_models(n_input_features=n_features, **kwargs)

            if prepend_parameter_keys_with_model is True:

                for m in models:

                    m.default_param_grid = {

                        f"{prepend_with}{k}": v for k, v in m.default_param_grid.items()

                    }

                    m.custom_param_grid = {

                        f"{prepend_with}{k}": v for k, v in m.custom_param_grid.items()

                    }

            if multi_output_only is True:

                models = [m for m in models if m.multi_output_Classifier is True]

            return models


### kernel_models

```python3
def kernel_models(
    svc: bool = True,
    svm: bool = False,
    **kwargs
) -> List[classical_ml_models.skmodel]
```

Function to build a default set of kernel models and parameter grids to optimize the models over

**Returns:**

| Type | Description |
|---|---|
| List[skmodel] | List of kernel models and meta data |

??? example "View Source"
        def kernel_models(svc: bool = True, svm: bool = False, **kwargs) -> List[skmodel]:

            """

            Function to build a default set of kernel models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of kernel models and meta data

            """

            models = []

            if svc is True or svm is True:

                # Support Vector Regression (SVR)

                models.append(

                    skmodel(

                        type(SVC()).__name__,

                        SVC(max_iter=10000),

                        {

                            "kernel": ["rbf", "linear"],

                            "C": [0.9, 1.0, 1.1],

                            "epsilon": [0.05, 0.1, 0.2],

                            "gamma": [None, 0.5, 1.5],

                        },

                        {

                            "kernel": ["catagorical", ["rbf", "linear"]],

                            "C": ["float", 0.01, 5.0],

                            "epsilon": ["float", 0.01, 1.0],

                            "gamma": ["float", 0.01, 5.0],

                        },

                        {},

                        False,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

                # Support Vector Regression (SVR) - polynomical kernel has the degree parameter so we separate the model

                models.append(

                    skmodel(

                        type(SVC()).__name__ + "poly_kernel",

                        SVC(max_iter=10000),

                        {

                            "kernel": ["poly"],

                            "C": [0.9, 1.0, 1.1],

                            "epsilon": [0.05, 0.1, 0.2],

                            "degree": [2, 3, 4, 5],

                            "gamma": [None, 0.5, 1.5],

                        },

                        {

                            "kernel": ["catagorical", ["poly"]],

                            "C": ["float", 0.01, 5.0],

                            "epsilon": ["float", 0.01, 1.0],

                            "degree": ["int", 2, 8],

                            "gamma": ["float", 0.01, 5.0],

                        },

                        {},

                        False,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            return models


### linear_models

```python3
def linear_models(
    logreg: bool = True,
    lr: bool = True,
    **kwargs
) -> List[classical_ml_models.skmodel]
```

Function to build a default set of linear models and parameter grids to optimize the models over

**Returns:**

| Type | Description |
|---|---|
| List[skmodel] | List of linear models and meta data |

??? example "View Source"
        def linear_models(logreg: bool = True, lr: bool = True, **kwargs) -> List[skmodel]:

            """

            Function to build a default set of linear models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of linear models and meta data

            """

            models = []

            if logreg is True or lr is True:

                # Linear regression

                models.append(

                    skmodel(

                        type(LogisticRegression()).__name__,

                        LogisticRegression(n_jobs=-1),

                        {

                            "penalty": ["l1", "l2", "elasticnet", "none"],

                            "C": [0.5, 1.0, 1.5],

                            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],

                            "fit_intercept": [True, False],

                        },

                        {

                            "penalty": ["catagorical", ["l1", "l2", "elasticnet", "none"]],

                            "C": ["float", [0.0, 1.5]],

                            "solver": [

                                "catagorical",

                                ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],

                            ],

                            "fit_intercept": ["catagorical", [True, False]],

                        },

                        {},

                        True,

                        "hhttps://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            return models


### neural_network_models

```python3
def neural_network_models(
    mlp: bool = True,
    layers: Tuple[Tuple[int, int, int], ...] = ((20, 10, 5), (50, 50, 50), (100, 50, 10), (100, 50, 25), (100, 50, 40)),
    solver: str = 'adam',
    max_iter: int = 400,
    n_input_features: Optional[int] = None,
    skip_feature_n_dependent: bool = False,
    **kwargs
) -> List[classical_ml_models.skmodel]
```

Function to build a default set of neural network models and parameter grids to optimize the models over

**Returns:**

| Type | Description |
|---|---|
| List[skmodel] | List of neural network models and meta data |

??? example "View Source"
        def neural_network_models(

            mlp: bool = True,

            layers: Tuple[Tuple[int, int, int], ...] = (

                (20, 10, 5),

                (50, 50, 50),

                (100, 50, 10),

                (100, 50, 25),

                (100, 50, 40),

            ),

            solver: str = "adam",

            max_iter: int = 400,

            n_input_features: Optional[int] = None,

            skip_feature_n_dependent: bool = False,

            **kwargs,

        ) -> List[skmodel]:

            """

             Function to build a default set of neural network models and parameter grids to optimize the models over

            Returns:

                List[skmodel]: List of neural network models and meta data

            """

            models = []

            if isinstance(layers, tuple):

                _layers = list(layers)

            if n_input_features is not None and skip_feature_n_dependent is False:

                if isinstance(layers, tuple):

                    _layers = list(layers)

                hl1 = [int(n_input_features * elt) for elt in [0.5, 0.7, 0.9]]

                hl2 = [int(n_input_features * elt) for elt in [0.3, 0.4, 0.5]]

                hl3 = [int(n_input_features * elt) for elt in [0.1, 0.2, 0.4]]

                for ith in range(3):

                    _layers.append((hl1[ith], hl2[ith], hl3[ith]))

            if mlp is True:

                # Multi-layer perceptron (MLP) - This is the only general NN in sklearn currently. It is built as feedforward network.

                models.append(

                    skmodel(

                        type(MLPClassifier()).__name__,

                        MLPClassifier(

                            max_iter=max_iter, solver=solver, random_state=utilities.random_seed

                        ),

                        {

                            "learning_rate": ["constant", "invscaling", "adaptive"],

                            "learning_rate_init": [0.0001, 0.001, 0.01],

                            "alpha": [0.00001, 0.0001, 0.0005, 0.001],

                            "activation": ["tanh", "relu"],

                            "hidden_layer_sizes": _layers,

                            "batch_size": ["auto", 16, 32],

                        },

                        {

                            "learning_rate": [

                                "catagorical",

                                ["constant", "invscaling", "adaptive"],

                            ],

                            "learning_rate_init": ["float", 0.0001, 0.2],

                            "alpha": ["float", 0.00001, 0.5],

                            "activation": [

                                "catagorical",

                                [

                                    "tanh",

                                    "relu",

                                ],

                            ],

                            "n_layers": ["int", 1, 5],

                            "batch_size": ["int", 10, 100],

                        },

                        {},

                        True,

                        "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier",

                        sklearn.__version__,

                        datetime.strftime(datetime.now(), "%d/%m/%Y"),

                    )

                )

            return models

## Classes

### skmodel

```python3
class skmodel(
    name: str,
    model: Union[Any, Callable],
    default_param_finite_grid: dict,
    default_param_range_priors: dict,
    custom_param_grid: dict,
    multi_output_Classifier: bool,
    more_info: Optional[str] = None,
    sklearn_version: str = '1.6.1',
    date: str = '01/05/2025'
)
```

skmodel(name: str, model: Union[Any, Callable], default_param_finite_grid: dict, default_param_range_priors: dict, custom_param_grid: dict, multi_output_Classifier: bool, more_info: Optional[str] = None, sklearn_version: str = '1.6.1', date: str = '01/05/2025')

#### Class variables

```python3
date
```

```python3
more_info
```

```python3
sklearn_version
```
