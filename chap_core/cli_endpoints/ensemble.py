# chap_core/cli_endpoints/ensemble.py

"""Ensemble evaluation commands for CHAP CLI."""

import logging
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from chap_core.api_types import BackTestParams, RunConfig
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import available_metrics
from chap_core.cli_endpoints._common import (
    create_model_lists,
    discover_geojson,
    load_dataset,
    load_dataset_from_csv,
    save_results,
)
from chap_core.database.model_templates_and_config_tables import (
    ConfiguredModelDB,
    ModelConfiguration,
    ModelTemplateDB,
)
from chap_core.ensemble.ensemble_model import EnsembleEstimator, NonNegativeMetaModel, ProbabilisticMetaModel
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate
from chap_core.models.utils import CHAP_RUNS_DIR

logger = logging.getLogger(__name__)


def evaluate_ensemble(
    base_model_names: Annotated[
        str,
        Parameter(
            help=(
                "Kommaseparert liste med base-modeller (lokale mapper eller GitHub-URLs). "
                "Eksempel: ../../../chap_modeller/minimalist_example_uv,"
                "../../../chap_modeller/rwanda_random_forest"
            )
        ),
    ],
    ensemble_method: Annotated[
        str,
        Parameter(
            help=(
                "Valg av ensemble-metode: 'deterministic' (NNLS + residualer) "
                "eller 'probabilistic' (CRPS-stacking av samples)."
            )
        ),
    ] = "probabilistic",
    # === datasett-argumenter, samme stil som evaluate() ===
    dataset_name: Annotated[
        str | None,
        Parameter(
            help=(
                "Navn på innebygd datasett (som i chap evaluate), f.eks. ISIMIP_dengue_harmonized. "
                "Hvis ikke satt, må --dataset-csv brukes."
            )
        ),
    ] = None,
    dataset_country: Annotated[
        str | None,
        Parameter(
            help=(
                "Land for multi-country datasett (f.eks. brazil for ISIMIP_dengue_harmonized). "
                "Kreves hvis dataset_name er multi-country."
            )
        ),
    ] = None,
    dataset_csv: Annotated[
        Path | None,
        Parameter(
            help=(
                "Path til CSV med disease data (time_period, location, disease_cases, "
                "og eventuelle kovariater). Brukes hvis dataset_name ikke er satt."
            )
        ),
    ] = None,
    polygons_json: Annotated[
        Path | None,
        Parameter(
            help=(
                "Optional: GeoJSON-fil med polygoner. Hvis ikke satt og dataset_csv brukes, forsøkes auto-discovery."
            ),
        ),
    ] = None,
    polygons_id_field: Annotated[
        str | None,
        Parameter(help="ID-felt i GeoJSON for locations (default: 'id')."),
    ] = "id",
    report_filename: Annotated[
        Path,
        Parameter(help="Basisnavn for rapport (uten .csv - det lages .csv og .i.csv filer)."),
    ] = Path("ensemble_report.csv"),
    output_file: Annotated[
        Path | None,
        Parameter(
            help=(
                "Path for output NetCDF file containing ensemble evaluation results "
                "(.nc extension). Hvis ikke satt, brukes report_filename med .nc-suffiks."
            )
        ),
    ] = None,
    backtest_params: Annotated[
        BackTestParams,
        Parameter(
            help=(
                "Backtest-konfigurasjon. "
                "Bruk --backtest-params.n-periods for prediksjonshorisont, "
                "--backtest-params.n-splits for antall train/test-splitt, "
                "--backtest-params.stride for steg mellom splittene."
            )
        ),
    ] = BackTestParams(n_periods=3, n_splits=7, stride=1),
    run_config: Annotated[
        RunConfig,
        Parameter(
            help=(
                "Model execution config. "
                "--run-config.is-chapkit-model, --run-config.debug, "
                "--run-config.ignore-environment, --run-config.run-directory-type."
            )
        ),
    ] = RunConfig(),
    model_configuration_yaml: Annotated[
        Path | None,
        Parameter(help="(Valgfritt) YAML med konfigurasjon for ALLE basemodeller (samme fil)."),
    ] = None,
    data_source_mapping: Annotated[
        Path | None,
        Parameter(
            help=(
                "Optional: JSON-fil som mapper modellens kovariatnavn til CSV-kolonnenavn. "
                'Format: {"rainfall": "precip_mm", ...}. Brukes bare hvis dataset_csv brukes.'
            )
        ),
    ] = None,
    historical_context_years: Annotated[
        int,
        Parameter(
            help=(
                "Antall år med historiske data som tas med som kontekst i evalueringen. "
                "Brukes til plotting/visualisering (samme som i `chap eval`)."
            )
        ),
    ] = 6,
):
    """
    Evaluér et stacked ensemble av flere CHAP-modeller.

    Eksempler:

        # Bruk innebygd datasett (samme som chap evaluate)
        chap evaluate-ensemble \\
            --base-model-names ../../../chap_modeller/minimalist_example_uv,../../../chap_modeller/rwanda_random_forest \\
            --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil \\
            --report-filename ensemble_report.csv

        # Bruk egen CSV
        chap evaluate-ensemble \\
            --base-model-names ../../../chap_modeller/minimalist_example_uv,../../../chap_modeller/rwanda_random_forest \\
            --dataset-csv ./data/your_dataset.csv \\
            --report-filename ensemble_report.csv
    """
    initialize_logging(run_config.debug, run_config.log_file)

    logger.info(f"Evaluating ensemble with base models: {base_model_names}")

    # 1) Last dataset på samme måte som evaluate()
    if dataset_name is not None:
        # Innebygd datasett
        dataset = load_dataset(
            dataset_country=dataset_country,
            dataset_csv=None,
            dataset_name=dataset_name,
            polygons_id_field=polygons_id_field,
            polygons_json=polygons_json,
        )
    else:
        # Må bruke dataset_csv
        assert dataset_csv is not None, "Må spesifisere enten --dataset-name eller --dataset-csv"
        column_mapping = None
        if data_source_mapping is not None:
            import json

            logger.info(f"Loading column mapping from {data_source_mapping}")
            with open(data_source_mapping) as f:
                column_mapping = json.load(f)

        geojson_path = polygons_json or discover_geojson(dataset_csv)
        dataset = load_dataset_from_csv(dataset_csv, geojson_path, column_mapping)

    # 2) Parse base_model_names til liste (gjenbruk _common.create_model_lists)
    _cfg_list_dummy, base_model_list = create_model_lists(
        model_configuration_yaml=None,
        model_name=base_model_names,
    )

    # 3) Last en (felles) ModelConfiguration hvis gitt
    configuration: ModelConfiguration | None = None
    if model_configuration_yaml is not None:
        import yaml

        logger.info(f"Loading model configuration from {model_configuration_yaml}")
        configuration = ModelConfiguration.model_validate(yaml.safe_load(open(model_configuration_yaml)))

    # 4) Lag ModelTemplates for alle base-modellene
    templates: list[ModelTemplate] = []
    for name in base_model_list:
        logger.info(f"Loading base model template from {name}")
        tpl = ModelTemplate.from_directory_or_github_url(
            name,
            base_working_dir=CHAP_RUNS_DIR,
            ignore_env=run_config.ignore_environment,
            run_dir_type=run_config.run_directory_type,
            is_chapkit_model=run_config.is_chapkit_model,
        )
        templates.append(tpl)

    # 5) Bygg EnsembleEstimator basert på valgt metode
    if ensemble_method == "probabilistic":
        # Probabilistisk meta-modell: CRPS-stacking av samples
        meta_model = ProbabilisticMetaModel(verbose=True)
        probabilistic_meta_model = True
        use_residual_bootstrap = False  # ingen residualer; meta-modellen bruker samples direkte
    elif ensemble_method == "deterministic":
        # Deterministisk meta-modell: NNLS på punktprediksjoner + residual-bootstrap
        meta_model = NonNegativeMetaModel()
        probabilistic_meta_model = False
        use_residual_bootstrap = True
    else:
        raise ValueError(f"ensemble_method må være 'deterministic' eller 'probabilistic', ikke {ensemble_method!r}")

    logger.info(f"Using ensemble_method = {ensemble_method}")

    ensemble_estimator = EnsembleEstimator(
        base_model_templates=templates,
        meta_model=meta_model,
        probabilistic_meta_model=probabilistic_meta_model,
        use_residual_bootstrap=use_residual_bootstrap,
        inner_val_periods=12,
    )

    # 6) Lag "syntetiske" DB-objekter for Evaluation
    model_template_db = ModelTemplateDB(
        id="ensemble_model",
        name="ensemble_model",
        version="0.1",
    )

    configured_model_db = ConfiguredModelDB(
        id="cli_eval_ensemble",
        model_template_id=model_template_db.id,
        model_template=model_template_db,
        configuration=configuration.model_dump() if configuration else {},
    )

    logger.info(
        f"Running ensemble backtest with {backtest_params.n_splits} splits, "
        f"{backtest_params.n_periods} periods, stride {backtest_params.stride}"
    )

    evaluation = Evaluation.create(
        configured_model=configured_model_db,
        estimator=ensemble_estimator,
        dataset=dataset,
        backtest_params=backtest_params,
        backtest_name="ensemble_evaluation",
        historical_context_years=historical_context_years,
    )

    # 6b) Lagre hele Evaluation til NetCDF for plotting / videre analyse
    if output_file is None:
        eval_nc = report_filename.with_suffix(".nc")
    else:
        eval_nc = output_file

    logger.info(f"Saved ensemble evaluation NetCDF to {eval_nc}")
    evaluation.to_file(str(eval_nc))

    # 6c) Logg vekter fra meta-modellen, hvis tilgjengelig
    weights = ensemble_estimator.weights
    if weights is not None:
        logger.info(f"Ensemble base model weights (percent): {weights}")
    else:
        logger.info("Ensemble base model weights not available (model not trained?).")

    # 7) Beregn globale metrics og lag CSV-rapporter
    flat = evaluation.to_flat()
    metrics_dict: dict[str, float] = {}

    for metric_id, metric_cls in available_metrics.items():
        metric = metric_cls()
        try:
            df_metric = metric.get_global_metric(flat.observations, flat.forecasts)
            if len(df_metric) == 1:
                metrics_dict[metric_id] = float(df_metric["metric"].iloc[0])
        except Exception as e:
            logger.warning(f"Failed to compute metric {metric_id}: {e}")

    # Gi modellnøkkelen et navn som inkluderer ensemble-metoden
    model_key = f"ensemble_{ensemble_method}"

    # (Valgfritt, men nyttig) legg navnet inn i metrics_dict også
    metrics_dict["model_name"] = model_key
    metrics_dict["ensemble_method"] = ensemble_method

    results_dict = {model_key: (metrics_dict, flat.forecasts)}
    save_results(str(report_filename), results_dict)

    return results_dict


def register_commands(app):
    """Register ensemble commands with the CLI app."""
    app.command(name="evaluate-ensemble")(evaluate_ensemble)
