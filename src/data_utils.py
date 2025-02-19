"""
data_utils.py: Funciones para carga y preprocesamiento de datos.
"""

# pylint: disable=invalid-name

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpieza de datos:
    Eliminación de columnas irrelevantes, tratamiento de valores nulos
    y codificación de variables categóricas.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    # Eliminamos características con multicolinealidad y la columna Id
    df.drop(['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'Id'],
            axis=1,
            inplace=True)

    # Eliminamos variables con muy pocos datos en más de una clase
    df.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'],
            axis=1,
            inplace=True)

    # Identificación de valores nulos
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    # Eliminamos variables con un alto porcentaje de valores nulos
    df.drop(['PoolQC', 'Alley', 'MiscFeature'], axis=1, inplace=True)

    # Tratamos las variables con valores faltantes
    df['Fence'] = df['Fence'].fillna('None')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    # Variables asociadas a Garage, Basement y Masonry
    df.drop(['GarageType', 'GarageFinish', 'GarageCond'], axis=1, inplace=True)
    df['GarageQual'] = df['GarageQual'].fillna('None')

    df.drop(['BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtFinType1'],
            axis=1,
            inplace=True)
    df['BsmtQual'] = df['BsmtQual'].fillna('None')
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    # Completamos valores faltantes con el valor más frecuente
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables categóricas a valores numéricos.

    Args:
        df (pd.DataFrame): DataFrame con columnas categóricas.

    Returns:
        pd.DataFrame: DataFrame con variables categóricas codificadas.
    """
    # Mapeo de variables categóricas ordinales
    quality_mapping = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'None': -1}
    df['ExterQual'] = df['ExterQual'].map(quality_mapping)
    df['BsmtQual'] = df['BsmtQual'].map(quality_mapping)
    df['KitchenQual'] = df['KitchenQual'].map(quality_mapping)

    # Codificación de variables categóricas con LabelEncoder
    categorical_features = df.select_dtypes(include=['object'])\
        .columns\
        .to_list()
    le = LabelEncoder()
    for col in categorical_features:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la selección de características usando SelectKBest.

    Args:
        df (pd.DataFrame): DataFrame limpio con características.

    Returns:
        pd.DataFrame: DataFrame con solo las mejores características.
    """
    features = df.drop(columns=['SalePrice'])
    target = df['SalePrice']

    selector = SelectKBest(score_func=f_regression, k=10)
    selected_features_array = selector.fit_transform(features, target)

    selected_feature_names = features.columns[selector.get_support()]
    df_selected = pd.DataFrame(selected_features_array,
                               columns=selected_feature_names)

    # Agregar la columna objetivo nuevamente
    df_selected['SalePrice'] = target.values

    return df_selected
