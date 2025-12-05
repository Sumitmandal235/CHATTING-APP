{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyOtjQGl40ZRYnbYBos0KCNi",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Sumitmandal235/CHATTING-APP/blob/main/Titanic_surver.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "j1eUxNrVwhl2",
        "outputId": "305a0abb-001d-4a96-c924-8bed28eeb70f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   PassengerId  Survived  Pclass  \\\n",
            "0            1         0       3   \n",
            "1            2         1       1   \n",
            "2            3         1       3   \n",
            "3            4         1       1   \n",
            "4            5         0       3   \n",
            "\n",
            "                                                Name     Sex   Age  SibSp  \\\n",
            "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
            "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
            "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
            "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
            "4                           Allen, Mr. William Henry    male  35.0      0   \n",
            "\n",
            "   Parch            Ticket     Fare Cabin Embarked  \n",
            "0      0         A/5 21171   7.2500   NaN        S  \n",
            "1      0          PC 17599  71.2833   C85        C  \n",
            "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
            "3      0            113803  53.1000  C123        S  \n",
            "4      0            373450   8.0500   NaN        S  \n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 891 entries, 0 to 890\n",
            "Data columns (total 12 columns):\n",
            " #   Column       Non-Null Count  Dtype  \n",
            "---  ------       --------------  -----  \n",
            " 0   PassengerId  891 non-null    int64  \n",
            " 1   Survived     891 non-null    int64  \n",
            " 2   Pclass       891 non-null    int64  \n",
            " 3   Name         891 non-null    object \n",
            " 4   Sex          891 non-null    object \n",
            " 5   Age          714 non-null    float64\n",
            " 6   SibSp        891 non-null    int64  \n",
            " 7   Parch        891 non-null    int64  \n",
            " 8   Ticket       891 non-null    object \n",
            " 9   Fare         891 non-null    float64\n",
            " 10  Cabin        204 non-null    object \n",
            " 11  Embarked     889 non-null    object \n",
            "dtypes: float64(2), int64(5), object(5)\n",
            "memory usage: 83.7+ KB\n",
            "None\n",
            "Accuracy on Training data: 0.8089887640449438\n",
            "Accuracy on Test data: 0.7821229050279329\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipython-input-2006543797.py:24: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
            "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
            "\n",
            "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
            "\n",
            "\n",
            "  data['Age'].fillna(data['Age'].mean(), inplace=True)\n",
            "/tmp/ipython-input-2006543797.py:27: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
            "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
            "\n",
            "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
            "\n",
            "\n",
            "  data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)\n",
            "/tmp/ipython-input-2006543797.py:31: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
            "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
            "\n",
            "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
            "\n",
            "\n",
            "  data['Fare'].fillna(data['Fare'].median(), inplace=True)\n",
            "/tmp/ipython-input-2006543797.py:35: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`\n",
            "  data.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix\n",
        "\n",
        "# 1. Load the Dataset\n",
        "# Ensure the file 'Titanic-Dataset.csv' is in your folder\n",
        "try:\n",
        "    data = pd.read_csv('/content/Titanic-Dataset.csv')\n",
        "except FileNotFoundError:\n",
        "    raise FileNotFoundError(\"Titanic-Dataset.csv not found in the current folder. Place the CSV next to this script.\")\n",
        "\n",
        "# 2. Data Exploration & Cleaning\n",
        "print(data.head())\n",
        "print(data.info())\n",
        "\n",
        "# Drop columns that likely won't help prediction (Name, Ticket, Cabin, PassengerId)\n",
        "data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)\n",
        "\n",
        "# Handle missing values: Fill missing Age with the mean age\n",
        "data['Age'].fillna(data['Age'].mean(), inplace=True)\n",
        "\n",
        "# Fill missing Embarked values with the mode (most common port)\n",
        "data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)\n",
        "\n",
        "# Fill missing Fare values (common source of errors)\n",
        "if 'Fare' in data.columns:\n",
        "    data['Fare'].fillna(data['Fare'].median(), inplace=True)\n",
        "\n",
        "# 3. Encoding Categorical Variables\n",
        "# Convert 'Sex' to numerical: Male=0, Female=1\n",
        "data.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)\n",
        "\n",
        "# Ensure all feature columns are numeric (coerce any lingering non-numeric)\n",
        "data = data.apply(pd.to_numeric, errors='coerce')\n",
        "\n",
        "# If coercion introduced NaNs, fill them (fallback)\n",
        "data.fillna(data.median(numeric_only=True), inplace=True)\n",
        "\n",
        "# 4. Splitting Features and Target\n",
        "X = data.drop(columns=['Survived'], axis=1) # Features (Age, Sex, Pclass, etc.)\n",
        "Y = data['Survived']                        # Target (0 = Dead, 1 = Survived)\n",
        "\n",
        "# Split into training and test data (80% train, 20% test)\n",
        "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)\n",
        "\n",
        "# 5. Model Training\n",
        "# increase max_iter to avoid convergence errors\n",
        "model = LogisticRegression(max_iter=1000)\n",
        "model.fit(X_train, Y_train)\n",
        "\n",
        "# 6. Evaluation\n",
        "# Prediction on training data\n",
        "train_prediction = model.predict(X_train)\n",
        "print(f\"Accuracy on Training data: {accuracy_score(Y_train, train_prediction)}\")\n",
        "\n",
        "# Prediction on test data\n",
        "test_prediction = model.predict(X_test)\n",
        "print(f\"Accuracy on Test data: {accuracy_score(Y_test, test_prediction)}\")"
      ]
    }
  ]
}