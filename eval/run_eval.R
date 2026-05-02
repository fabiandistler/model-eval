# Runs the `are` eval for models listed in data/models.yaml
# Will skip models that have already been run (by looking in results_rds)
# Combines all rds results into data/data_combined.rds
#
# Usage:
#   Rscript eval/run_eval.R                    # Run all unevaluated models
#   Rscript eval/run_eval.R --model minimax_m2_7  # Run a single model

library(ellmer)
library(vitals)
library(purrr)
library(glue)

# Source helper functions
source(here::here("R/task_definition.R"))
source(here::here("R/data_loading.R"))
source(here::here("R/eval_functions.R"))

# Configuration
YAML_PATH <- here::here("data/models.yaml")
RESULTS_DIR <- here::here("results_rds")
LOG_DIR <- here::here("logs")
SCORER_MODEL <- "claude-3-7-sonnet-latest"

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
selected_model <- NULL
if ("--model" %in% args) {
  idx <- which(args == "--model")
  if (idx < length(args)) {
    selected_model <- args[idx + 1]
    message(glue("CLI flag --model detected; will run only: {selected_model}"))
  } else {
    stop("--model requires a model_id argument")
  }
}

# ---------------------------------------------------------------------------
# Set up logging (redirect messages to file + console)
# ---------------------------------------------------------------------------
if (!dir.exists(LOG_DIR)) {
  dir.create(LOG_DIR, recursive = TRUE)
}
log_file <- file.path(LOG_DIR, format(Sys.time(), "eval_%Y%m%d_%H%M%S.log"))
con <- file(log_file, open = "wt")
sink(con, split = TRUE) # split = TRUE writes to both file and console
message(glue("Log file: {log_file}\n"))

# Set up vitals logging
vitals::vitals_log_dir_set(LOG_DIR)

# ============================================================================
# Run Evaluation
# ============================================================================

# Parse YAML configuration
model_configs <- parse_model_configs(YAML_PATH)

# Find unevaluated models (optionally restricted to CLI selection)
selected_ids <- if (!is.null(selected_model)) selected_model else NULL
unevaluated <- find_unevaluated_models(model_configs, RESULTS_DIR, selected_ids = selected_ids)

# Run evaluations if needed
if (length(unevaluated) > 0) {
  message(glue("Running {length(unevaluated)} unevaluated model(s)..."))

  message("Initializing scorer chat...")
  scorer_chat <- chat_anthropic(model = SCORER_MODEL)
  message("Scorer chat initialized.")

  eval_results <- run_all_evals(
    model_configs = model_configs,
    unevaluated_ids = unevaluated,
    model_eval_fn = model_eval,
    results_dir = RESULTS_DIR,
    scorer_chat = scorer_chat
  )

  # Report failures only
  n_failed <- sum(!eval_results)
  if (n_failed > 0) {
    message(glue("\nWarning: {n_failed} model(s) failed"))
    failed_ids <- names(eval_results)[!eval_results]
    walk(failed_ids, ~ message(glue("  - {model_configs[[.x]]$name}")))
  }
} else {
  message("No unevaluated models to run.")
}

# Combine results
combine_results(
  yaml_path = YAML_PATH,
  results_dir = RESULTS_DIR,
  load_model_info_fn = load_model_info,
  load_eval_results_fn = load_eval_results,
  process_eval_data_fn = process_eval_data,
  compute_cost_data_fn = compute_cost_data
)

# Close log sink
sink()
close(con)
message(glue("\nLog saved to: {log_file}"))
