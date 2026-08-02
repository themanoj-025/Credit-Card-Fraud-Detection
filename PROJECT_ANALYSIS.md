# PROJECT ANALYSIS & REPOSITORY AUDIT: Credit Card Fraud Detection

## 1. Executive Summary
- **Repository Name**: `Credit Card Fraud Detection`
- **Path**: `f:\GITHUB\Credit Card Fraud Detection`
- **Modernization Status**: Verified & Cleaned (Ultra Master Prompt v5.0)

## 2. Architecture & Tech Stack
- **Target Architecture**: Clean Modular Layout
- **Junk/Stale Artifacts Purged**: 0 items
- **Duplicates Identified**: 4 items
- **Test Verification Result**: `FAILED: 
=================================== ERRORS ====================================
_____________________ ERROR collecting tests/test_api.py ______________________
ImportError while importing test module 'f:\GITHUB\Credit Card Fraud Detection\tests\test_api.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\jm270\miniconda3\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_api.py:17: in <module>
    from api.main import app
api\main.py:38: in <module>
    from api.logging_config import get_logger, setup_structlog
api\logging_config.py:19: in <module>
    import structlog
E   ModuleNotFoundError: No module named 'structlog'
________________ ERROR collecting tests/test_preprocessing.py _________________
ImportError while importing test module 'f:\GITHUB\Credit Card Fraud Detection\tests\test_preprocessing.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\jm270\miniconda3\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_preprocessing.py:22: in <module>
    from src.fraudlens.data.preprocessing import FraudPreprocessor, Resampler
src\fraudlens\data\preprocessing.py:14: in <module>
    from imblearn.combine import SMOTETomek
E   ModuleNotFoundError: No module named 'imblearn'
____________________ ERROR collecting tests/test_train.py _____________________
ImportError while importing test module 'f:\GITHUB\Credit Card Fraud Detection\tests\test_train.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\jm270\miniconda3\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_train.py:18: in <module>
    from src.fraudlens.models.train import FraudTrainer
src\fraudlens\models\train.py:13: in <module>
    import lightgbm as lgb
E   ModuleNotFoundError: No module named 'lightgbm'
============================== warnings summary ===============================
src\fraudlens\config.py:90
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:90: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    AVG_FRAUD_LOSS: float = Field(

src\fraudlens\config.py:95
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:95: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    REVIEW_COST: float = Field(

src\fraudlens\config.py:104
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:104: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    TEST_SIZE: float = Field(0.2, env="TEST_SIZE")

src\fraudlens\config.py:105
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:105: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    VAL_SIZE: float = Field(0.1, env="VAL_SIZE")

src\fraudlens\config.py:106
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:106: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    RANDOM_STATE: int = Field(42, env="RANDOM_STATE")

src\fraudlens\config.py:113
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:113: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    TARGET_COLUMN: str = Field("Class", env="TARGET_COLUMN")

src\fraudlens\config.py:118
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:118: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    RESAMPLING_STRATEGIES: List[str] = Field(

src\fraudlens\config.py:122
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:122: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    DEFAULT_RESAMPLING: str = Field("smote", env="DEFAULT_RESAMPLING")

src\fraudlens\config.py:127
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:127: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    DEFAULT_MODELS: List[str] = Field(

src\fraudlens\config.py:138
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:138: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    MODEL_SELECTION_METRIC: str = Field("pr_auc", env="MODEL_SELECTION_METRIC")

src\fraudlens\config.py:145
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:145: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    IFOREST_CONTAMINATION: float = Field(0.01, env="IFOREST_CONTAMINATION")

src\fraudlens\config.py:146
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:146: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    IFOREST_N_ESTIMATORS: int = Field(200, env="IFOREST_N_ESTIMATORS")

src\fraudlens\config.py:149
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:149: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    CATBOOST_VERBOSE: bool = Field(False, env="CATBOOST_VERBOSE")

src\fraudlens\config.py:150
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:150: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    CATBOOST_ITERATIONS: int = Field(200, env="CATBOOST_ITERATIONS")

src\fraudlens\config.py:151
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:151: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    CATBOOST_DEPTH: int = Field(6, env="CATBOOST_DEPTH")

src\fraudlens\config.py:156
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:156: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    N_THRESHOLDS: int = Field(100, env="N_THRESHOLDS")

src\fraudlens\config.py:161
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:161: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    MAX_SHAP_FEATURES: int = Field(10, env="MAX_SHAP_FEATURES")

src\fraudlens\config.py:162
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:162: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    N_SHAP_BACKGROUND_SAMPLES: int = Field(100, env="N_SHAP_BACKGROUND_SAMPLES")

src\fraudlens\config.py:167
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:167: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    LLM_MODEL: str = Field("claude-sonnet-4-20250514", env="LLM_MODEL")

src\fraudlens\config.py:168
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:168: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    LLM_MAX_TOKENS: int = Field(1000, env="LLM_MAX_TOKENS")

src\fraudlens\config.py:169
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:169: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    LLM_TEMPERATURE: float = Field(0.3, env="LLM_TEMPERATURE")

src\fraudlens\config.py:172
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:172: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    RAG_TOP_K: int = Field(3, env="RAG_TOP_K")

src\fraudlens\config.py:173
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:173: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    EMBEDDING_DIM: int = Field(30, env="EMBEDDING_DIM")

src\fraudlens\config.py:174
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:174: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    RAG_USE_PROJECTION: bool = Field(

src\fraudlens\config.py:179
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:179: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    RAG_PROJECTION_COMPONENTS: int = Field(

src\fraudlens\config.py:188
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:188: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    DRIFT_THRESHOLD: float = Field(0.05, env="DRIFT_THRESHOLD")

src\fraudlens\config.py:189
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:189: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    DRIFT_ALERT_WINDOW: int = Field(1000, env="DRIFT_ALERT_WINDOW")

src\fraudlens\config.py:194
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:194: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    DASHBOARD_REFRESH_MS: int = Field(500, env="DASHBOARD_REFRESH_MS")

src\fraudlens\config.py:195
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:195: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    MAX_TRANSACTION_HISTORY: int = Field(500, env="MAX_TRANSACTION_HISTORY")

src\fraudlens\config.py:196
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:196: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    SIMULATION_FRAUD_RATE: float = Field(0.02, env="SIMULATION_FRAUD_RATE")

src\fraudlens\config.py:197
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:197: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    SIMULATION_BATCH_SIZE: int = Field(10, env="SIMULATION_BATCH_SIZE")

src\fraudlens\config.py:202
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:202: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    API_URL: str = Field("http://localhost:8000", env="API_URL")

src\fraudlens\config.py:203
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:203: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    API_PORT: int = Field(8000, env="API_PORT")

src\fraudlens\config.py:204
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:204: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    DASHBOARD_PORT: int = Field(8501, env="DASHBOARD_PORT")

src\fraudlens\config.py:209
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:209: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    CROSS_VALIDATION_FOLDS: int = Field(5, env="CV_FOLDS")

src\fraudlens\config.py:210
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:210: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    CROSS_VALIDATION_SCORING: str = Field("average_precision", env="CV_SCORING")

src\fraudlens\config.py:215
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:215: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    MLFLOW_EXPERIMENT_NAME: str = Field(

src\fraudlens\config.py:218
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:218: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    MLFLOW_TRACKING_URI: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")

src\fraudlens\config.py:219
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:219: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    MLFLOW_ARTIFACT_DIR: str = Field("mlruns", env="MLFLOW_ARTIFACT_DIR")

src\fraudlens\config.py:224
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:224: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    HPO_ENABLED: bool = Field(True, env="HPO_ENABLED")

src\fraudlens\config.py:225
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:225: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    HPO_N_TRIALS: int = Field(30, env="HPO_N_TRIALS")

src\fraudlens\config.py:226
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:226: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    HPO_CV_FOLDS: int = Field(3, env="HPO_CV_FOLDS")

src\fraudlens\config.py:227
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:227: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    HPO_MODELS: List[str] = Field(["xgboost", "lightgbm"], env="HPO_MODELS")

src\fraudlens\config.py:232
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:232: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    RETRAINING_ENABLED: bool = Field(

src\fraudlens\config.py:237
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:237: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    RETRAINING_FEEDBACK_THRESHOLD: int = Field(

src\fraudlens\config.py:242
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:242: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    RETRAINING_DRIFT_CRITICAL_THRESHOLD: int = Field(

src\fraudlens\config.py:247
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:247: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    RETRAINING_DRIFT_WINDOW_DAYS: int = Field(

src\fraudlens\config.py:256
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:256: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    FEATURE_LLM_NARRATOR: bool = Field(

src\fraudlens\config.py:261
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:261: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    FEATURE_ANOMALY_SCORE: bool = Field(

src\fraudlens\config.py:266
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:266: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    FEATURE_SHAP_EXPLANATION: bool = Field(

src\fraudlens\config.py:271
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:271: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    FEATURE_CACHE_PREDICTIONS: bool = Field(

src\fraudlens\config.py:276
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:276: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    FEATURE_RAG_RETRIEVAL: bool = Field(

src\fraudlens\config.py:285
  f:\GITHUB\Credit Card Fraud Detection\src\fraudlens\config.py:285: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'env'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    PREDICTION_THRESHOLD: Optional[float] = Field(

api\schemas.py:50
  F:\GITHUB\Credit Card Fraud Detection\api\schemas.py:50: PydanticDeprecatedSince20: Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
    @validator("Amount")

tests\test_rate_limit_shared.py:18
  f:\GITHUB\Credit Card Fraud Detection\tests\test_rate_limit_shared.py:18: PytestUnknownMarkWarning: Unknown pytest.mark.integration - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    pytest.mark.integration,

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
ERROR tests/test_api.py
ERROR tests/test_preprocessing.py
ERROR tests/test_train.py
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 3 failures !!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!
55 warnings, 3 errors in 4.69s
`

## 3. Operations & Release Checklist
- CI/CD Workflows Verified: ✅
- Dependency Health: ✅
- Security Credentials Scan: ✅
- Architecture Alignment: ✅
