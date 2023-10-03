"""APE Wrapper for common data science python functions."""

import json
from typing import Dict, Optional, Tuple, cast

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from bs4 import BeautifulSoup
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats  # type: ignore
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.experimental import (enable_halving_search_cv,
                                  enable_iterative_imputer)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (ElasticNet, LinearRegression,
                                  LogisticRegression, Perceptron, Ridge)
from sklearn.metrics import \
    classification_report as classification_report_sklearn
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.model_selection import cross_val_score as cross_val_score_sklearn
from sklearn.model_selection import \
    train_test_split as train_test_split_sklearn
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree as plot_tree_sklearn
from textblob import TextBlob
from wordcloud import WordCloud

sns.set()


# =============================================================================
#                            Type Wrappers
# =============================================================================

# Custom type wrapper (annotations + intellisense)
class NumberColumn(str):
    """Decorative wrapper for APE type matching"""

class StrColumn(str):
    """Decorative wrapper for APE type matching"""

class BoolColumn(str):
    """Decorative wrapper for APE type matching"""

class DateTimeColumn(str):
    """Decorative wrapper for APE type matching"""

class NumberSeries(pd.Series):
    """Decorative wrapper for APE type matching"""

class StrSeries(pd.Series):
    """Decorative wrapper for APE type matching"""

class BoolSeries(pd.Series):
    """Decorative wrapper for APE type matching"""

class DateTimeSeries(pd.Series):
    """Decorative wrapper for APE type matching"""

class NumberDataFrame(pd.DataFrame):
    """Decorative wrapper for APE type matching"""

class EmbeddingMatrix(np.ndarray):
    """Decorative wrapper for APE type matching"""


# =============================================================================
#                            Data Loading
# =============================================================================

def load_table_csv(table_path: str, **kwargs) -> pd.DataFrame:
    """Loads csv file into pandas `DataFrame`.

    Args:
        table_path (str): Path to file as string.

    Returns:
        pd.DataFrame: Loaded table.
    """
    df = pd.read_csv(table_path, **kwargs)
    display(df.info())
    display(df.head())
    return df


# =============================================================================
#                            Type Conversion
# =============================================================================

def asfloat(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
) -> NumberSeries:
    """casts type NOT INPLACE -> all labels are lost"""
    if col is None or isinstance(data, pd.Series):
        return cast(NumberSeries, data.astype(float))
    return cast(NumberSeries, data[col].astype(float))


def asint(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
) -> NumberSeries:
    """casts type NOT INPLACE -> all labels are lost"""
    if col is None or isinstance(data, pd.Series):
        return cast(NumberSeries, data.astype(int))
    return cast(NumberSeries, data[col].astype(int))


def asDateTime(
    data: pd.DataFrame | StrSeries,
    col: Optional[NumberColumn]=None,
) -> DateTimeSeries:
    """casts type NOT INPLACE -> all labels are lost"""
    if col is None or isinstance(data, pd.Series):
        return cast(DateTimeSeries, pd.to_datetime(data))
    return cast(DateTimeSeries, pd.to_datetime(data[col]))


# =============================================================================
#                            Descriptive Statistics
# =============================================================================

def describe(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
    **kwargs,
) -> pd.DataFrame | pd.Series:
    """Descriptive statistics for a `Series`, `DataFrame` or `DataFrame[Column]`.

    Args:
        data (pd.DataFrame | pd.Series): Some tabular data.
        col (Optional[str], optional): Column of table. Defaults to `None`.

    Returns:
        pd.DataFrame | pd.Series: `Dataframe` if data is a `Series` or no column was given, else `Series`.
    """
    if col is None or isinstance(data, pd.Series):
        describe_res = data.describe(**kwargs)
    else:
        describe_res = data[col].describe(**kwargs)
    display(describe_res)
    return describe_res


def value_counts(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
) -> NumberSeries:
    """Value counts of a `Series`, `DataFrame` or `DataFrame[Column]`.

    Args:
        data (pd.DataFrame | pd.Series): Some tabular data.
        col (Optional[str], optional): Column of table. Defaults to `None`.

    Returns:
        NumberSeries: Value counts.
    """
    if col is None or isinstance(data, pd.Series):
        value_counts_res = data.value_counts()
    else:
        value_counts_res = data[col].value_counts()
    display(value_counts_res)
    return cast(NumberSeries, value_counts_res)


def skew(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
    **kwargs,
) -> NumberSeries | float:
    """Skew of numeric column or series.
    """
    if col is None or isinstance(data, pd.Series):
        res = data.skew(**kwargs)
    else:
        res = data[col].skew(**kwargs)
    display(res)
    return cast(float, res)


def kurt(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
    **kwargs,
) -> NumberSeries | float:
    """Kurtosis of numeric column or series.
    """
    if col is None or isinstance(data, pd.Series):
        res = data.kurt(**kwargs)
    else:
        res = data[col].kurt(**kwargs)
    display(res)
    return cast(float, res)


def corr(
    data: pd.DataFrame,
    col1: Optional[NumberColumn]=None,
    col2: Optional[NumberColumn]=None,
    **kwargs,
) -> pd.DataFrame | float:
    """Correlation of all numeric columns to eachother (`DataFrame`)
    or of two specific columns (`float`).
    """
    if col1 is None or col2 is None:
        res = data.corr(numeric_only=True, **kwargs)
    else:
        res = data[col1].corr(data[col2], **kwargs)
    display(res)
    return res


def k_most_corr_indep_var_corr_matrix(
    data: pd.DataFrame,
    col: NumberColumn,
    k: int,
) -> pd.DataFrame:
    """Matrix of `k` most to `col` correlating columns."""
    corr_df = data.corr(numeric_only=True)
    n_most = corr_df[col].abs().nlargest(k).index
    return corr_df.loc[n_most, n_most] # type: ignore


# =============================================================================
#                            Plotting
# =============================================================================

# -----------------------------------------------------------------------------
#                            Distribution Plots
# -----------------------------------------------------------------------------

def histplot(
    df: pd.DataFrame,
    col: NumberColumn,
    hue: Optional[str]=None,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """> kwarg `hue` should be column with few features."""
    if not fig or not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    return fig, sns.histplot(data=df, x=col, kde=True, hue=hue, ax=ax, **kwargs)


def boxplot(
    df: pd.DataFrame,
    y: NumberColumn,
    x: Optional[str]=None,
    hue: Optional[str]=None,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """> kwarg `x` should be column with few features.
    > kwarg `hue` should be column with few features."""
    if not fig or not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    return fig, sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)


def jointplot(
    df: pd.DataFrame,
    x: NumberColumn,
    y: NumberColumn,
    hue: Optional[str]=None,
    **kwargs,
) -> Figure:
    """> kwarg `hue` should be column with few features."""
    return sns.jointplot(data=df, x=x, y=y, hue=hue, **kwargs)


def kdeplot(
    df: pd.DataFrame,
    col: NumberColumn,
    hue: Optional[str]=None,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """> kwarg `hue` should be column with few features."""
    if not fig or not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    return fig, sns.kdeplot(data=df, x=col, hue=hue, ax=ax, **kwargs)


def countplot(
    df: pd.DataFrame,
    col: str,
    hue: Optional[str]=None,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """> kwarg `hue` should be column with few features."""
    if not fig or not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    return fig, sns.countplot(data=df, x=col, hue=hue, ax=ax, **kwargs)


def normality_plots(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    if not fig or not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    if col is not None and isinstance(data, pd.DataFrame):
        ax = sns.histplot(data=data, x=col, kde=True, stat='density', ax=ax, **kwargs)
        mu, std = stats.norm.fit(data[col])
    else:
        ax = sns.histplot(data=data, kde=True, stat='density', ax=ax, **kwargs)
        mu, std = stats.norm.fit(data)
    xx = np.linspace(*ax.get_xlim(), 100) # type: ignore
    ax.plot(xx, stats.norm.pdf(xx, mu, std), 'r') # type: ignore
    plt.subplots(nrows=1, ncols=1)
    if col is not None and isinstance(data, pd.DataFrame):
        _ = stats.probplot(data[col], plot=plt)
    else:
        _ = stats.probplot(data, plot=plt)
    return fig, ax # type: ignore


# -----------------------------------------------------------------------------
#                            Relationship Plots
# -----------------------------------------------------------------------------

def scatterplot(
    df: pd.DataFrame,
    x: NumberColumn,
    y: NumberColumn,
    hue: Optional[str]=None,
    style: Optional[str]=None,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """> kwarg `hue` should be column with few features.
    > kwarg `style` should be column with few features.
    """
    if not fig or not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    return fig, sns.scatterplot(data=df, x=x, y=y, hue=hue, style=style, ax=ax, **kwargs)


def heatmap(
    df: pd.DataFrame,
    piv_col1: Optional[str]=None,
    piv_col2: Optional[str]=None,
    num_col: Optional[NumberColumn]=None,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """`piv_col1`, `piv_col2` and `num_col`
    can be used to pivot the table before creating a heatmap"""
    if all((piv_col1, piv_col2, num_col)):
        df = df.pivot(index=piv_col1, columns=piv_col2, values=num_col)
    if not fig or not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
    return fig, sns.heatmap(df, ax=ax, annot=df.shape[0]<11, **kwargs)


def barplot(
    df: pd.DataFrame,
    x: str,
    y: NumberColumn,
    hue: Optional[str]=None,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """> arg `x` and kwarg `hue` should be column with few features."""
    if not fig or not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    return fig, sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)


# -----------------------------------------------------------------------------
#                            Trend Plots
# -----------------------------------------------------------------------------

def lineplot(
    df: pd.DataFrame,
    x: NumberColumn,
    y: NumberColumn,
    hue: Optional[str]=None,
    style: Optional[str]=None,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """> kwarg `hue` should be column with few features.
    > kwarg `style` should be column with few features.
    """
    if not fig or not ax:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    return fig, sns.lineplot(data=df, x=x, y=y, hue=hue, style=style, ax=ax, **kwargs)


# -----------------------------------------------------------------------------
#                            Multi Plots
# -----------------------------------------------------------------------------

def pairplot(
    data: pd.DataFrame,
    col: Optional[NumberColumn]=None,
    n: Optional[int]=None,
    hue: Optional[str]=None,
    **kwargs,
) -> Figure:
    """Passing `col` and `n` will only display `n` most correlating features.
    > kwarg `hue` should be column with few features."""
    if col is not None and n is not None:
        corr = data.corr(numeric_only=True)
        n_most = corr[col].abs().nlargest(n).index # type: ignore
        grid = sns.pairplot(data[n_most], hue=hue, **kwargs) # type: ignore
    else:
        grid = sns.pairplot(data, hue=hue, **kwargs)
    return grid.figure


# -----------------------------------------------------------------------------
#                            Utility
# -----------------------------------------------------------------------------

def rotate_x_labels(figure: Figure, axes: Axes, **kwargs) -> Tuple[Figure, Axes]:
    axes.tick_params('x', labelrotation=90, **kwargs)
    display(figure)
    return figure, axes


def set_figure_size(x_size: int, y_size: int) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(x_size, y_size))
    plt.close()
    return fig, ax


# =============================================================================
#                            Index Manipulation
# =============================================================================

def select_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Simple wrapper for APE typing"""
    return df[col]


def drop(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.drop(columns=col)

def drop_i(df: pd.DataFrame, col: str) -> None:
    df.drop(columns=col, inplace=True)
    return


# =============================================================================
#                            Numerical Transformations
# =============================================================================

def log(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
) -> pd.DataFrame | NumberSeries:
    """> This transformation is **NOT** inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            copy = data.copy()
            numeric = copy.select_dtypes(include=np.number).columns # type: ignore
            copy.loc[:, numeric] = copy[numeric].applymap(np.log)
            return copy
        data = data[col] # type: ignore
    return np.log(data) # type: ignore


def log_i(data: pd.DataFrame | NumberSeries, col: Optional[NumberColumn]=None) -> None:
    """> This transformation is inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            numeric = data.select_dtypes(include=np.number).columns # type: ignore
            data.loc[:, numeric] = data[numeric].applymap(np.log)
            return
        data[col] = np.log(data[col])
        return
    data.loc[:] = np.log(data) # type: ignore
    return


def log1p(data: pd.DataFrame | NumberSeries, col: Optional[NumberColumn]=None) -> pd.DataFrame | NumberSeries:
    """> This transformation is **NOT** inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            copy = data.copy()
            numeric = copy.select_dtypes(include=np.number).columns # type: ignore
            copy.loc[:, numeric] = copy[numeric].applymap(np.log1p)
            return copy
        data = data[col] # type: ignore
    return np.log1p(data) # type: ignore


def log1p_i(data: pd.DataFrame | NumberSeries, col: Optional[NumberColumn]=None) -> None:
    """> This transformation is inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            numeric = data.select_dtypes(include=np.number).columns # type: ignore
            data.loc[:, numeric] = data[numeric].applymap(np.log1p)
            return
        data[col] = np.log1p(data[col])
        return
    data.loc[:] = np.log1p(data) # type: ignore
    return


def boxcox(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
) -> pd.DataFrame | NumberSeries:
    """> This transformation is **NOT** inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            copy = data.copy()
            numeric = copy.select_dtypes(include=np.number).columns # type: ignore
            copy.loc[:, numeric] = copy[numeric].apply(
                lambda s: pd.Series(stats.boxcox(s)[0], index=s.index),
                axis=0,
            )
            return copy
        data = data[col] # type: ignore
    return pd.Series(stats.boxcox(data)[0], data.index) # type: ignore


def boxcox_i(data: pd.DataFrame | NumberSeries, col: Optional[NumberColumn]=None) -> None:
    """> This transformation is inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            numeric = data.select_dtypes(include=np.number).columns # type: ignore
            data.loc[:, numeric] = data[numeric].apply(
                lambda s: pd.Series(stats.boxcox(s)[0], index=s.index),
                axis=0,
            )
            return
        data[col] = pd.Series(stats.boxcox(data[col]), data.index)
        return
    data.loc[:] = pd.Series(stats.boxcox(data)[0], data.index) # type: ignore
    return


def z_score(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
) -> pd.DataFrame | NumberSeries:
    """> This transformation is **NOT** inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            copy = data.copy()
            numeric = copy.select_dtypes(include=np.number).columns # type: ignore
            copy.loc[:, numeric] = copy[numeric].apply(
                lambda s: (s - s.mean()) / s.std(),
                axis=0,
            )
            return copy
        data = data[col] # type: ignore
    return (data - data.mean()) / data.std() # type: ignore


# =============================================================================
#                            Normalization
# =============================================================================

def num_z_score(df: pd.DataFrame) -> pd.DataFrame:
    """Simple wrapper for z_score"""
    return cast(pd.DataFrame, z_score(data=df))


def z_score_i(data: pd.DataFrame | NumberSeries, col: Optional[NumberColumn]=None) -> None:
    """> This transformation is inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            numeric = data.select_dtypes(include=np.number).columns # type: ignore
            data.loc[:, numeric] = data[numeric].apply(lambda s: (s-s.mean())/s.std(), axis=0)
            return
        data[col] = (data[col] - data[col].mean()) / data[col].std()
        return
    data.loc[:] = (data - data.mean()) / data.std() # type: ignore
    return


def num_z_score_i(df: pd.DataFrame) -> None:
    """Simple wrapper for z_score_i"""
    return z_score_i(data=df)


def center_mean(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
) -> pd.DataFrame | NumberSeries:
    """> This transformation is **NOT** inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            copy = data.copy()
            numeric = copy.select_dtypes(include=np.number).columns # type: ignore
            copy.loc[:, numeric] = copy[numeric].apply(lambda s: s - s.mean(), axis=0)
            return copy
        data = data[col] # type: ignore
    return data - data.mean() # type: ignore


def num_center_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Simple wrapper for center_mean"""
    return cast(pd.DataFrame, center_mean(data=df))


def center_mean_i(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
) -> None:
    """> This transformation is inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            numeric = data.select_dtypes(include=np.number).columns # type: ignore
            data.loc[:, numeric] = data[numeric].apply(lambda s: s-s.mean(), axis=0)
            return
        data[col] = data[col] - data[col].mean()
        return
    data.loc[:] = data - data.mean() # type: ignore
    return


def num_center_mean_i(df: pd.DataFrame) -> None:
    """Simple wrapper for center_mean_i"""
    return center_mean_i(data=df)


def min_max_norm(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
) -> pd.DataFrame | NumberSeries:
    """> This transformation is **NOT** inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            copy = data.copy()
            numeric = copy.select_dtypes(include=np.number).columns # type: ignore
            copy.loc[:, numeric] = copy[numeric].apply(
                lambda s: (s-s.min())/(s.max()-s.min()),
                axis=0,
            )
            return copy
        data = data[col] # type: ignore
    return (data + data.min()) / (data.max() - data.min()) # type: ignore


def num_min_max_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Simple wrapper for min_max_norm"""
    return cast(pd.DataFrame, min_max_norm(data=df))


def min_max_norm_i(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
) -> None:
    """> This transformation is inplace"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            numeric = data.select_dtypes(include=np.number).columns # type: ignore
            data.loc[:, numeric] = data[numeric].apply(
                lambda s: (s-s.min())/(s.max()-s.min()),
                axis=0,
            )
            return
        data[col] = (data[col] + data[col].min()) / (data[col].max() - data[col].min())
        return
    data.loc[:] = (data + data.min()) / (data.max() - data.min()) # type: ignore
    return

def num_min_max_norm_i(df: pd.DataFrame) -> pd.DataFrame:
    """Simple wrapper for min_max_norm_i"""
    return cast(pd.DataFrame, min_max_norm_i(data=df))


# =============================================================================
#                            String Transformations
# =============================================================================

def replace_re(
    data: pd.DataFrame | StrSeries,
    col: Optional[StrColumn]=None,
    pattern: str='',
    replacer: str='',
    **kwargs,
) -> StrSeries:
    if isinstance(data, pd.DataFrame) and col is not None:
        copy = data[col].copy()
    else:
        copy = cast(StrSeries, data).copy()
    return cast(StrSeries, copy.str.replace(pat=pattern, repl=replacer, **kwargs))


def replace_re_i(
    data: pd.DataFrame | StrSeries,
    col: Optional[StrColumn]=None,
    pattern: str='',
    replacer: str='',
    **kwargs,
) -> None:
    if isinstance(data, pd.DataFrame) and col is not None:
        data.loc[:, col] = data[col].str.replace(pat=pattern, repl=replacer, **kwargs)
        return
    data.loc[:] = data.str.replace(pat=pattern, repl=replacer, **kwargs) # type: ignore


def strip(
    data: pd.DataFrame | StrSeries,
    col: Optional[StrColumn]=None,
) -> StrSeries:
    if isinstance(data, pd.DataFrame) and col is not None:
        copy = data[col].copy()
    else:
        copy = cast(StrSeries, data).copy()
    return cast(StrSeries, copy.str.strip())


def strip_i(
    data: pd.DataFrame | StrSeries,
    col: Optional[StrColumn]=None,
) -> None:
    if isinstance(data, pd.DataFrame) and col is not None:
        data.loc[:, col] = data[col].str.strip()
        return
    data.loc[:] = data.str.strip() # type: ignore


def extract(
    data: pd.DataFrame | StrSeries,
    col: Optional[StrColumn]=None,
    pattern: str='',
    **kwargs,
) -> StrSeries:
    if isinstance(data, pd.DataFrame) and col is not None:
        copy = data[col].copy()
    else:
        copy = cast(StrSeries, data).copy()
    return cast(StrSeries, copy.str.extract(pat=pattern, expand=False, **kwargs))


def extract_i(
    data: pd.DataFrame | StrSeries,
    col: Optional[StrColumn]=None,
    pattern: str='',
    **kwargs,
) -> None:
    if isinstance(data, pd.DataFrame) and col is not None:
        data.loc[:, col] = data[col].str.extract(pat=pattern, expand=False, **kwargs)
        return
    data.loc[:] = data.str.extract(pat=pattern, expand=False, **kwargs) # type: ignore


def match(
    data: pd.DataFrame | StrSeries,
    col: Optional[StrColumn]=None,
    pattern: str='',
    **kwargs,
) -> BoolSeries:
    """>There is NO inplace version of this function"""
    if isinstance(data, pd.DataFrame) and col is not None:
        copy = data[col].copy()
    else:
        copy = cast(StrSeries, data).copy()
    return cast(BoolSeries, copy.str.match(pat=pattern, **kwargs))


def lower(
    data: pd.DataFrame | StrSeries,
    col: Optional[StrColumn]=None,
) -> StrSeries:
    if isinstance(data, pd.DataFrame) and col is not None:
        copy = data[col].copy()
    else:
        copy = cast(StrSeries, data).copy()
    return cast(StrSeries, copy.str.lower())


def lower_i(
    data: pd.DataFrame | StrSeries,
    col: Optional[StrColumn]=None,
) -> None:
    if isinstance(data, pd.DataFrame) and col is not None:
        data.loc[:, col] = data[col].str.lower()
        return
    data.loc[:] = data.str.lower() # type: ignore


# -----------------------------------------------------------------------------
#                            Categorical / Nominal Transformations
# -----------------------------------------------------------------------------

def one_hot_encode(
    data: pd.DataFrame | pd.Series,
    col: Optional[StrColumn]=None,
) -> NumberDataFrame:
    """Encodes categorical column into binary columns"""
    if isinstance(data, pd.DataFrame) and col is not None:
        data = data[col]
    return pd.get_dummies(data, drop_first=True)


def one_hot_encode_i(
    data: pd.DataFrame,
    col: StrColumn,
) -> None:
    """Encodes categorical column into binary columns
    and replaces old column with new ones.
    """
    new_cols = pd.get_dummies(data[col], drop_first=True)
    data.drop(columns=col, inplace=True)
    data[new_cols.columns] = new_cols


def bin_nominal(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
    bins: int=10,
) -> Tuple[pd.Series, pd.Series]:
    """Bins nominal column into `bins` bins
    and returns binned numerical series and bins.

    > Consider using `bin_nominal_q` instead to bin into quantiles.
    """
    if isinstance(data, pd.DataFrame) and col is not None:
        data = data[col]
    out, bins = pd.cut(cast(pd.Series, data), bins=bins, retbins=True, labels=False)
    return out, pd.Series(bins)


def bin_nominal_i(
    data: pd.DataFrame | NumberSeries,
    col: NumberColumn,
    bins: int=10,
) -> pd.Series:
    """Bins nominal column into `bins` bins,
    replaces old column with binned one and returns bins.

    > Consider using `bin_nominal_q_i` instead to bin into quantiles.
    """
    if isinstance(data, pd.DataFrame) and col is not None:
        out, bins = bin_nominal(data=data, col=col, bins=bins)
        data[col] = out
    elif isinstance(data, pd.Series):
        out, bins = bin_nominal(data=data, bins=bins)
        data.loc[:] = out
    else:
        raise TypeError('Dataframe but no column given')
    return bins


def bin_nominal_q(
    data: pd.DataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
    bins: int=10,
) -> Tuple[pd.Series, pd.Series]:
    """Bins nominal column into `bins` bins
    and returns binned numerical series and bins.
    """
    if isinstance(data, pd.DataFrame) and col is not None:
        data = data[col]
    out, bins = pd.qcut(cast(pd.Series, data), q=bins, retbins=True, labels=False)
    return out, pd.Series(bins)


def bin_nominal_q_i(
    data: pd.DataFrame | NumberSeries,
    col: NumberColumn,
    bins: int=10,
) -> pd.Series:
    """Bins nominal column into `bins` bins,
    replaces old column with binned one and returns bins.
    """
    if isinstance(data, pd.DataFrame) and col is not None:
        out, bins = bin_nominal_q(data=data, col=col, bins=bins)
        data[col] = out
    elif isinstance(data, pd.Series):
        out, bins = bin_nominal_q(data=data, bins=bins)
        data.loc[:] = out
    else:
        raise TypeError('Dataframe but no column given')
    return bins


# =============================================================================
#                            Reshaping / Sorting
# =============================================================================

def pivot(df: pd.DataFrame, new_columns: str, new_index: str, new_values: str) -> pd.DataFrame:
    """Pivot with no value aggregation -> NO DUPLICATE VALUE HANDLING"""
    return df.pivot(columns=new_columns, index=new_index, values=new_values)


def pivot_table_mean(
    df: pd.DataFrame,
    new_columns: str,
    new_index: str,
    new_values: NumberColumn,
    **kwargs,
) -> NumberDataFrame:
    """Pivot WITH value aggregation `np.mean`"""
    return cast(NumberDataFrame, df.pivot_table(
        columns=new_columns,
        index=new_index,
        values=new_values,
        aggfunc=np.mean,
        **kwargs,
    ))


def drop_duplicates(
    data: pd.DataFrame | pd.Series,
    col: Optional[pd.Series | str]=None,
    **kwargs,
) -> pd.DataFrame | pd.Series | Tuple[pd.DataFrame, pd.Series]:
    """Modes:
    1. DataFrame -> DataFrame: removes all duplicate rows
    2. DataFrame, Str -> DataFrame: removes all rows with duplcate entries in Col
    3. DataFrame, Series -> DataFrame, Series:
    removes all rows with duplicate entries in Series in both the DataFrame and the Series
    4. Series -> Series: removes all duplcate entries in Series
    """
    if isinstance(data, pd.DataFrame):
        if col is None:
            return data.drop_duplicates(**kwargs)
        if isinstance(col, pd.Series):
            series_copy = col.drop_duplicates(**kwargs)
            return data.loc[series_copy.index], series_copy
        return data.drop_duplicates(subset=col, **kwargs)
    return data.drop_duplicates(**kwargs)


def drop_duplicates_i(
    data: pd.DataFrame | pd.Series,
    col: Optional[pd.Series | str]=None,
    **kwargs,
) -> None:
    """Modes:
    1. DataFrame -> : removes all duplicate rows
    2. DataFrame, Str -> : removes all rows with duplcate entries in Col
    3. DataFrame, Series -> :
    removes all rows with duplicate entries in Series in both the DataFrame and the Series
    4. Series -> : removes all duplcate entries in Series
    """
    if isinstance(data, pd.DataFrame):
        if col is None:
            data.drop_duplicates(inplace=True, **kwargs)
            return
        if isinstance(col, pd.Series):
            duplicated = col.duplicated(**kwargs).replace(False, np.nan).dropna().index
            col.drop(index=duplicated, inplace=True)
            data.drop(index=duplicated, inplace=True)
            return
        data.drop_duplicates(subset=col, inplace=True, **kwargs)
        return
    data.drop_duplicates(inplace=True, **kwargs)


# =============================================================================
#                            Outliers
# =============================================================================

def filter_sd(
    data: pd.DataFrame,
    col: NumberColumn,
    abs_sd: Optional[int | float]=None,
    **kwargs,
) -> pd.DataFrame:
    """Selects entries where masked feature is an outlier
    by at least `abs_sd` standard deviations. Defaults to 2.
    """
    df = data.loc[
        abs(data[col] - data[col].mean()) > (abs_sd if abs_sd else 2) * data[col].std(**kwargs)
    ]
    display(df)
    return df


def drop_sd(
    data: pd.DataFrame,
    col: NumberColumn,
    abs_sd: Optional[int | float]=None
) -> pd.DataFrame:
    """Removes entries where masked feature is an outlier
    by at least `abs_sd` standard deviations. Defaults to 2.
    """
    return data.loc[
        abs(data[col] - data[col].mean()) <= (abs_sd if abs_sd else 2) * data[col].std()
    ]


def drop_sd_i(
    data: pd.DataFrame,
    col: NumberColumn,
    abs_sd: Optional[int | float]=None
) -> None:
    """Removes entries where masked feature is an outlier
    by at least `abs_sd` standard deviations. Defaults to 2.
    """
    outliers = data.loc[
        abs(data[col] - data[col].mean()) > (abs_sd if abs_sd else 2) * data[col].std()
    ].index
    data.drop(index=outliers, inplace=True)


# =============================================================================
#                            Missing Values
# =============================================================================

def dropna(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
    **kwargs,
) -> pd.DataFrame | pd.Series:
    if col is None:
        return data.dropna(**kwargs)
    return cast(pd.DataFrame, data)[col].dropna(**kwargs)


def dropna_i(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
    **kwargs,
) -> None:
    if col is None:
        data.dropna(inplace=True, **kwargs)
        return
    cast(pd.DataFrame, data)[col].dropna(inplace=True, **kwargs)


def dropna_col(
    data: pd.DataFrame
) -> pd.DataFrame | pd.Series:
    return data.dropna(axis=1)


def dropna_col_i(
    data: pd.DataFrame
) -> None:
    data.dropna(inplace=True, axis=1)
    return


def na_count(df: pd.DataFrame, n: Optional[int]=None) -> NumberSeries:
    series = cast(NumberSeries, (df.isna().sum() / df.shape[0]))
    display(series.sort_values(ascending=False).head(n if n else 30))
    return series


def na_count_percentage(df: pd.DataFrame, n: Optional[int]=None) -> NumberSeries:
    series = cast(NumberSeries, df.isna().sum() / df.shape[0])
    display(series.sort_values(ascending=False).head(n if n else 30))
    return series


def isna(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
) -> pd.DataFrame | pd.Series:
    if col is None:
        return data.isna()
    return cast(pd.DataFrame, data)[col].isna()


def notna(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
) -> pd.DataFrame | pd.Series:
    if col is None:
        return data.notna()
    return cast(pd.DataFrame, data)[col].notna()


def impute_mean(
    data: NumberDataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
    **kwargs,
) -> NumberDataFrame | NumberSeries:
    """Imputes missing values with the mean of the column"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            return data.fillna(data.mean(), **kwargs)
        return data[col].fillna(data[col].mean(), **kwargs)
    return data.fillna(data.mean(), **kwargs)


def impute_mean_i(
    data: NumberDataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
    **kwargs,
) -> None:
    """Imputes missing values with the mean of the column"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            data.fillna(data.mean(), inplace=True, **kwargs)
            return
        data[col].fillna(data[col].mean(), inplace=True, **kwargs)
        return
    data.fillna(data.mean(), inplace=True, **kwargs)


def impute_median(
    data: NumberDataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
    **kwargs,
) -> NumberDataFrame | NumberSeries:
    """Imputes missing values with the median of the column"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            return data.fillna(data.median(), **kwargs)
        return data[col].fillna(data[col].median(), **kwargs)
    return data.fillna(data.median(), **kwargs)


def impute_median_i(
    data: NumberDataFrame | NumberSeries,
    col: Optional[NumberColumn]=None,
    **kwargs,
) -> None:
    """Imputes missing values with the median of the column"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            data.fillna(data.median(), inplace=True, **kwargs)
            return
        data[col].fillna(data[col].median(), inplace=True, **kwargs)
        return
    data.fillna(data.median(), inplace=True, **kwargs)


def impute_mode(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
    **kwargs,
) -> pd.DataFrame | pd.Series:
    """Imputes missing values with the mode of the column"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            return data.fillna(data.mode(), **kwargs)
        return data[col].fillna(data[col].mode(), **kwargs)
    return data.fillna(data.mode(), **kwargs)


def impute_mode_i(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
    **kwargs,
) -> None:
    """Imputes missing values with the mode of the column"""
    if isinstance(data, pd.DataFrame):
        if col is None:
            data.fillna(data.mode(), inplace=True, **kwargs)
            return
        data[col].fillna(data[col].mode(), inplace=True, **kwargs)
        return
    data.fillna(data.mode(), inplace=True, **kwargs)


def impute_random_sample(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
    **kwargs,
) -> pd.DataFrame | pd.Series:
    """Imputes missing values with a random sample from the column"""
    data_copy = data.copy()
    if isinstance(data_copy, pd.DataFrame):
        cols = [col] if col else data_copy.columns
        for column in cols:
            missing_values_count = data_copy[column].isnull().sum()
            if missing_values_count > 0:
                data_copy[column] = data_copy[column].fillna(data_copy[column].dropna().sample(
                    n=missing_values_count,
                    replace=True,
                    **kwargs,
                ))
    else:
        data_copy = data_copy.fillna(data_copy.dropna().sample(
            n=data_copy.isnull().sum(),
            replace=True,
            **kwargs,
        ))
    return data_copy


def impute_random_sample_i(
    data: pd.DataFrame | pd.Series,
    col: Optional[str]=None,
    **kwargs,
) -> None:
    """Imputes missing values with a random sample from the column"""
    if isinstance(data, pd.DataFrame):
        cols = [col] if col else data.columns
        for column in cols:
            missing_values_count = data[column].isnull().sum()
            if missing_values_count > 0:
                data[column].fillna(data[column].dropna().sample(
                    n=missing_values_count,
                    replace=True,
                    **kwargs,
                ), inplace=True)
    else:
        data.fillna(data.dropna().sample(
            n=data.isnull().sum(),
            replace=True,
            **kwargs,
        ), inplace=True)


# =============================================================================
#                            Modeling
# =============================================================================

# DBSCAN | KMeans | PCA | TruncatedSVD | DummyClassifier | DummyRegressor | AdaBoostClassifier | AdaBoostRegressor | RandomForestClassifier | RandomForestRegressor | VotingClassifier | VotingRegressor | ElasticNet | LinearRegression | LogisticRegression | Perceptron | Ridge | DecisionTreeClassifier | DecisionTreeRegressor | KNeighborsClassifier | KNeighborsRegressor | LinearSVC | LinearSVR | KernelRidge | GridSearchCV | HalvingGridSearchCV | IterativeImputer | KNNImputer | SimpleImputer

# -----------------------------------------------------------------------------
#                            Utility
# -----------------------------------------------------------------------------

def init_sklearn_estimator(
    estimator: str,
    **kwargs,
) -> DBSCAN | KMeans | PCA | TruncatedSVD | DummyClassifier | DummyRegressor | AdaBoostClassifier | AdaBoostRegressor | RandomForestClassifier | RandomForestRegressor | VotingClassifier | VotingRegressor | ElasticNet | LinearRegression | LogisticRegression | Perceptron | Ridge | DecisionTreeClassifier | DecisionTreeRegressor | KNeighborsClassifier | KNeighborsRegressor | LinearSVC | LinearSVR | KernelRidge | GridSearchCV | HalvingGridSearchCV | IterativeImputer | KNNImputer | SimpleImputer:
    """Initializes a sklearn estimator.

    The passed string must be one of the following:

    - 'KernelRidgeRegressor'
    - 'PerceptronClassifier'
    - 'LogisticRegressionClassifier'
    - 'LinearRegressor'
    - 'ElasticNetRegressor'
    - 'RidgeRegressor'
    - 'DecisionTreeClassifier'
    - 'DecisionTreeRegressor'
    - 'LinearSVClassifier'
    - 'LinearSVRregressor'
    - 'RandomForestClassifier'
    - 'AdaBoostClassifier'
    - 'VotingClassifier'
    - 'RandomForestRegressor'
    - 'AdaBoostRegressor'
    - 'VotingRegressor'
    - 'DummyClassifier'
    - 'DummyRegressor'
    - 'KMeansClustor'
    - 'DBScanClustor'
    - 'KNeighborsClassifier'
    - 'KNeighborsRegressor'
    - 'GridSearchCV'
    - 'HalvingGridSearchCV'
    - 'SimpleImputer'
    - 'IterativeImputer'
    - 'KNNImputer'
    - 'KNNImputer'
    - 'CatKNNImputer' #! NO
    - 'PCA'
    - 'TruncatedSVD'
    """
    estimator_mapping: Dict[
        str,
        DBSCAN | KMeans | PCA | TruncatedSVD | DummyClassifier | DummyRegressor | AdaBoostClassifier | AdaBoostRegressor | RandomForestClassifier | RandomForestRegressor | VotingClassifier | VotingRegressor | ElasticNet | LinearRegression | LogisticRegression | Perceptron | Ridge | DecisionTreeClassifier | DecisionTreeRegressor | KNeighborsClassifier | KNeighborsRegressor | LinearSVC | LinearSVR | KernelRidge | GridSearchCV | HalvingGridSearchCV | IterativeImputer | KNNImputer | SimpleImputer
    ] = {
        'KernelRidgeRegressor': KernelRidge,
        'PerceptronClassifier': Perceptron,
        'LogisticRegressionClassifier': LogisticRegression,
        'LinearRegressor': LinearRegression,
        'ElasticNetRegressor': ElasticNet,
        'RidgeRegressor': Ridge,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'LinearSVClassifier': LinearSVC,
        'LinearSVRregressor': LinearSVR,
        'RandomForestClassifier': RandomForestClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,
        'VotingClassifier': VotingClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'VotingRegressor': VotingRegressor,
        'DummyClassifier': DummyClassifier,
        'DummyRegressor': DummyRegressor,
        'KMeansClustor': KMeans,
        'DBScanClustor': DBSCAN,
        'KNeighborsClassifier': KNeighborsClassifier,
        'KNeighborsRegressor': KNeighborsRegressor,
        'GridSearchCV': GridSearchCV,
        'HalvingGridSearchCV': HalvingGridSearchCV,
        'SimpleImputer': SimpleImputer,
        'IterativeImputer': IterativeImputer,
        'KNNImputer': KNNImputer,
        # 'CatKNNImputer': None,  #! maybe
        'PCA': PCA,
        'TrunctuatedSVD': TruncatedSVD,
    }
    return estimator_mapping[estimator](**kwargs)


def init_sklearn_search_cv(
    search_cv: str,
    estimator: str,
    **kwargs,
) -> GridSearchCV | HalvingGridSearchCV:
    """Initializes a sklearn search cv.

    The passed search cv string must be one of the following:

    - 'GridSearchCV'
    - 'HalvingGridSearchCV'

    The passed estimator string must be an estimator (see `init_sklearn_estimator`)

    > **MISSING** paramter `param_grid` has to be provided here:
        Dict[str, List] or List[Dict[str, List]
    """
    search_cv_mapping: Dict[
        str,
        GridSearchCV | HalvingGridSearchCV
    ] = {
        'GridSearchCV': GridSearchCV,
        'HalvingGridSearchCV': HalvingGridSearchCV,
    }
    return search_cv_mapping[search_cv](
        estimator=init_sklearn_estimator(estimator),
        **kwargs,
    )


def init_sklearn_voting_estimator(
    voting_estimator: str,
    estimator_list: str,
    **kwargs,
) -> VotingClassifier | VotingRegressor:
    """Initializes a sklearn voting estimator.

    The passed voting estimator string must be one of the following:

    - 'VotingClassifier'
    - 'VotingRegressor'

    The passed estimator list string must be a comma seperated list of estimators
    (see `init_sklearn_estimator`)
    """
    voting_estimator_mapping: Dict[
        str,
        VotingClassifier | VotingRegressor
    ] = {
        'VotingClassifier': VotingClassifier,
        'VotingRegressor': VotingRegressor,
    }
    return voting_estimator_mapping[voting_estimator](
        estimators=[
            (estimator, init_sklearn_estimator(estimator))
            for estimator
            in estimator_list.split(',')
        ],
        **kwargs,
    )


def plot_tree(
    clf: DecisionTreeClassifier | DecisionTreeRegressor,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a FITTED decision tree"""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(20, 20))
    plot_tree_sklearn(clf, ax=ax, **kwargs)
    return fig, ax


def plot_tree_importances(
    clf: DecisionTreeClassifier | DecisionTreeRegressor | RandomForestClassifier | RandomForestRegressor,
    X: pd.DataFrame,
    fig: Optional[Figure]=None,
    ax: Optional[Axes]=None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots feature importances for a FITTED decision tree or random forest"""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(20, 20))
    feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
    ax = sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax, **kwargs)
    return fig, ax


def column_split(
    df: pd.DataFrame,
    column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits a dataframe into X and y based on a column name"""
    X = df.drop(columns=column)
    y = df[column]
    return X, y


def column_split_i(
    df: pd.DataFrame,
    column: str,
) -> Tuple[pd.Series]:
    """Splits a dataframe into X and y *inplace* based on a column name
    > **WARNING** This will modify the passed dataframe
    """
    y = df[column]
    df.drop(columns=column, inplace=True)
    return y


def train_test_split(
    df: pd.DataFrame,
    y: pd.Series | str,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.Series | str, pd.DataFrame, pd.Series | str]:
    """Splits a dataframe into X_train, y_train, X_test, y_test
    > returns strings instead of series if y is a string
    """
    if isinstance(y, str):
        X, y = column_split(df, y)
    else:
        X = df
    X_train, X_test, y_train, y_test = train_test_split_sklearn(X, y, **kwargs)
    if y.name in df.columns:
        X_train[y.name] = y_train
        y_train = y.name
        X_test[y.name] = y_test
        y_test = y.name
    return X_train, y_train, X_test, y_test

# -----------------------------------------------------------------------------
#                            Text Processing
# -----------------------------------------------------------------------------

def plot_wordcloud(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
    **kwargs,
) -> Figure:
    """Plots a wordcloud for a dataframe or series"""
    if isinstance(data, pd.DataFrame):
        data = data[column]
    figure = plt.figure(figsize=(10, 10))
    wordcloud = WordCloud(background_color='white', **kwargs).generate(' '.join(data))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return figure


def embed_text_word2vec(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
    word2vec: Optional[gensim.models.Word2Vec]=None,
    **kwargs,
) -> Tuple[EmbeddingMatrix, gensim.models.Word2Vec] | EmbeddingMatrix:
    """Trains a word2vec model on a dataframe or series and returns the embeddings and the model.
    Alternatively, pass a pretrained model as the word2vec argument.
    """
    if isinstance(data, pd.DataFrame):
        data = data[column]
    if word2vec is None:
        word2vec_model = gensim.models.Word2Vec(
            data,
            **kwargs,
        )
    else:
        word2vec_model = word2vec
    embeddings = np.array([
        np.mean([
            word2vec_model.wv[word]
            for word
            in sequence
            if word in word2vec_model.wv
        ], axis=0)
        for sequence
        in data
    ])
    if word2vec is None:
        return embeddings, word2vec_model
    return embeddings


def embed_text_tfidf(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
    tfidf: Optional[TfidfVectorizer]=None,
    **kwargs,
) -> Tuple[EmbeddingMatrix, TfidfVectorizer] | EmbeddingMatrix:
    """Trains a tfidf vectorizer on a dataframe or series
    and returns the embeddings and the vectorizer.
    > pass ngrams as a tuple of (min_n, max_n).
    Alternatively, pass a pretrained vectorizer as the tfidf argument.
    """
    if isinstance(data, pd.DataFrame):
        data = data[column]
    if tfidf is None:
        tfidf_vectorizer = TfidfVectorizer(
            **kwargs,
        )
        tfidf_vectorizer.fit(data)
    else:
        tfidf_vectorizer = tfidf
    embeddings = tfidf_vectorizer.transform(data).toarray()
    if tfidf is None:
        return embeddings, tfidf_vectorizer
    return embeddings


def embed_text_bow(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
    bow: Optional[CountVectorizer]=None,
    **kwargs,
) -> Tuple[EmbeddingMatrix, CountVectorizer] | EmbeddingMatrix:
    """Trains a bag-of-words vectorizer on a dataframe or series
    and returns the embeddings and the vectorizer.
    > pass ngrams as a tuple of (min_n, max_n).
    Alternatively, pass a pretrained vectorizer as the bow argument.
    """
    if isinstance(data, pd.DataFrame):
        data = data[column]
    if bow is None:
        bow_vectorizer = CountVectorizer(
            **kwargs,
        )
        bow_vectorizer.fit(data)
    else:
        bow_vectorizer = bow
    embeddings = bow_vectorizer.transform(data).toarray()
    if bow is None:
        return embeddings, bow_vectorizer
    return embeddings


def get_text_from_html(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> pd.Series:
    """Extracts text from html using BeautifulSoup"""
    if isinstance(data, pd.DataFrame):
        data = data[column]
    return data.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())


def get_text_from_html_i(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> None:
    """Extracts text from html using BeautifulSoup *inplace*"""
    if isinstance(data, pd.DataFrame) and column is not None:
        data[column] = data[column].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    elif isinstance(data, pd.Series):
        data[:] = data.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())


def remove_urls(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> pd.Series:
    """Removes urls from text"""
    if isinstance(data, pd.DataFrame):
        data = data[column]
    return data.str.replace(r'http\S+', '', regex=True)


def remove_urls_i(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> None:
    """Removes urls from text *inplace*"""
    if isinstance(data, pd.DataFrame) and column is not None:
        data[column] = data[column].str.replace(r'http\S+', '', regex=True)
    elif isinstance(data, pd.Series):
        data[:] = data.str.replace(r'http\S+', '', regex=True)


def fix_spelling(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> pd.Series:
    """Fixes spelling using textblob"""
    if isinstance(data, pd.DataFrame):
        data = data[column]
    return data.apply(lambda x: str(TextBlob(x).correct()))


def fix_spelling_i(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> None:
    """Fixes spelling using textblob *inplace*"""
    if isinstance(data, pd.DataFrame) and column is not None:
        data[column] = data[column].apply(lambda x: str(TextBlob(x).correct()))
    elif isinstance(data, pd.Series):
        data[:] = data.apply(lambda x: str(TextBlob(x).correct()))


def remove_stopwords(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> pd.Series:
    """Removes stopwords using spacy"""
    nlp = spacy.load('en_core_web_sm')
    if isinstance(data, pd.DataFrame):
        data = data[column]


def remove_stopwords_i(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> None:
    """Removes stopwords using spacy *inplace*"""
    nlp = spacy.load('en_core_web_sm')
    if isinstance(data, pd.DataFrame) and column is not None:
        data[column] = data[column].apply(lambda x: ' '.join([token.text for token in nlp(x) if not token.is_stop]))
    elif isinstance(data, pd.Series):
        data[:] = data.apply(lambda x: ' '.join([token.text for token in nlp(x) if not token.is_stop]))


def lemmatize(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> pd.Series:
    """Lemmatizes using spacy"""
    nlp = spacy.load('en_core_web_sm')
    if isinstance(data, pd.DataFrame):
        data = data[column]
    return data.apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))


def lemmatize_i(
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> None:
    """Lemmatizes using spacy *inplace*"""
    nlp = spacy.load('en_core_web_sm')
    if isinstance(data, pd.DataFrame) and column is not None:
        data[column] = data[column].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))
    elif isinstance(data, pd.Series):
        data[:] = data.apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))


def expand_abbr(
    path_to_dict: str,
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> pd.Series:
    """Expands abbreviations using a dictionary loaded from a json file"""
    if isinstance(data, pd.DataFrame):
        data = data[column]
    with open(path_to_dict, 'r') as f:
        abbr_dict = json.load(f)
    return data.apply(lambda x: ' '.join(
        abbr_dict[word] if word in abbr_dict else word
        for word
        in x.split()
    ))


def expand_abbr_i(
    path_to_dict: str,
    data: pd.DataFrame | pd.Series,
    column: Optional[str]=None,
) -> None:
    """Expands abbreviations using a dictionary loaded from a json file *inplace*"""
    with open(path_to_dict, 'r') as f:
        abbr_dict = json.load(f)
    if isinstance(data, pd.DataFrame) and column is not None:
        data[column] = data[column].apply(lambda x: ' '.join(
            abbr_dict[word] if word in abbr_dict else word
            for word
            in x.split()
        ))
    elif isinstance(data, pd.Series):
        data[:] = data.apply(lambda x: ' '.join(
            abbr_dict[word] if word in abbr_dict else word
            for word
            in x.split()
        ))


# -----------------------------------------------------------------------------
#                            Fitting, Transforming, Predicting
# -----------------------------------------------------------------------------

def fit_estimator(
    estimator: DBSCAN | KMeans | DummyClassifier | DummyRegressor | AdaBoostClassifier | AdaBoostRegressor | RandomForestClassifier | RandomForestRegressor | VotingClassifier | VotingRegressor | ElasticNet | LinearRegression | LogisticRegression | Perceptron | Ridge | DecisionTreeClassifier | DecisionTreeRegressor | KNeighborsClassifier | KNeighborsRegressor | LinearSVC | LinearSVR | KernelRidge | GridSearchCV | HalvingGridSearchCV,
    X: pd.DataFrame | EmbeddingMatrix,
    y: Optional[pd.Series | str]=None,
    **kwargs,
) -> DBSCAN | KMeans | DummyClassifier | DummyRegressor | AdaBoostClassifier | AdaBoostRegressor | RandomForestClassifier | RandomForestRegressor | VotingClassifier | VotingRegressor | ElasticNet | LinearRegression | LogisticRegression | Perceptron | Ridge | DecisionTreeClassifier | DecisionTreeRegressor | KNeighborsClassifier | KNeighborsRegressor | LinearSVC | LinearSVR | KernelRidge | GridSearchCV | HalvingGridSearchCV:
    """Fits an estimator
    > Operation is in-place even though it returns the estimator!
    """
    if isinstance(y, str):
        y = X[y]
        X = X.drop(columns=y)
    return estimator.fit(X, y, **kwargs)


def predict(
    estimator: DBSCAN | KMeans | DummyClassifier | DummyRegressor | AdaBoostClassifier | AdaBoostRegressor | RandomForestClassifier | RandomForestRegressor | VotingClassifier | VotingRegressor | ElasticNet | LinearRegression | LogisticRegression | Perceptron | Ridge | DecisionTreeClassifier | DecisionTreeRegressor | KNeighborsClassifier | KNeighborsRegressor | LinearSVC | LinearSVR | KernelRidge | GridSearchCV | HalvingGridSearchCV,
    X: pd.DataFrame | EmbeddingMatrix,
    **kwargs,
) -> pd.Series:
    """Predicts using a FITTED estimator"""
    try:
        return pd.Series(estimator.predict(X, **kwargs), index=X.index, name='prediction')
    except AttributeError:
        return pd.Series(estimator.predict(X, **kwargs), name='prediction')


def fit_transformer(
    transformer: SimpleImputer | IterativeImputer | KNNImputer | PCA | TruncatedSVD,
    X: pd.DataFrame,
) -> SimpleImputer | IterativeImputer | KNNImputer | PCA | TruncatedSVD:
    """Fits a transformer
    > Operation is in-place even though it returns the transformer!
    """
    return transformer.fit(X)


def transform(
    transformer: SimpleImputer | IterativeImputer | KNNImputer | PCA | TruncatedSVD | TfidfVectorizer | CountVectorizer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Transforms using a FITTED transformer
    > May completly change schema if anything other than imputation is done.
    """
    res = transformer.transform(X)
    if isinstance(transformer, (SimpleImputer, IterativeImputer, KNNImputer)):
        return pd.DataFrame(res, columns=X.columns, index=X.index)
    return pd.DataFrame(res, index=X.index)


def fit_transform(
    transformer: SimpleImputer | IterativeImputer | KNNImputer | PCA | TruncatedSVD,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Fits and transforms using a transformer
    > May completly change schema if anything other than imputation is done.
    """
    transformer.fit(X)
    return transform(transformer, X)


def transform_i(
    transformer: SimpleImputer | IterativeImputer | KNNImputer,
    X: pd.DataFrame,
) -> None:
    """Imputes using a FITTED impute transformer in-place
    """
    X[:] = transform(transformer, X)


def fit_transform_i(
    transformer: SimpleImputer | IterativeImputer | KNNImputer,
    X: pd.DataFrame,
) -> None:
    """Fits and imputes in-place
    """
    transformer.fit(X)
    transform_i(transformer, X)

# -----------------------------------------------------------------------------
#                            Model Evalutation
# -----------------------------------------------------------------------------

def score_metric(
    y_true: pd.Series,
    y_pred: pd.Series,
    metric: str,
) -> float:
    """Scores a metric on a prediction"""
    scorer = get_scorer(metric)
    score = scorer(y_true, y_pred)
    display(score)
    return score


def classification_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    **kwargs,
) -> pd.DataFrame:
    """Displays a classification report"""
    report = pd.DataFrame(
        classification_report_sklearn(y_true, y_pred, output_dict=True, **kwargs)
    ).transpose()
    display(report)
    return report


def cross_val_score(
    estimator: DBSCAN | KMeans | DummyClassifier | DummyRegressor | AdaBoostClassifier | AdaBoostRegressor | RandomForestClassifier | RandomForestRegressor | VotingClassifier | VotingRegressor | ElasticNet | LinearRegression | LogisticRegression | Perceptron | Ridge | DecisionTreeClassifier | DecisionTreeRegressor | KNeighborsClassifier | KNeighborsRegressor | LinearSVC | LinearSVR | KernelRidge | GridSearchCV | HalvingGridSearchCV,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str,
    **kwargs,
) -> NumberSeries:
    """Cross validates a model"""
    scorer = get_scorer(metric)
    scores = cross_val_score_sklearn(estimator, X, y, scoring=scorer, **kwargs)
    display(scores)
    return pd.Series(scores, name='scores')
