# This file defines:
# - create_are_task(): Function to create the evaluation task
# - model_eval(): Core function to evaluate a single model

library(ellmer)
library(vitals)

# Set results directory
results_dir <- here::here("results_rds")

#' Create the ARE evaluation task
#'
#' @param scorer_chat Chat object used for model-graded scoring
#' @return A Task object configured for ARE evaluation
create_are_task <- function(scorer_chat) {
  Task$new(
    dataset = are,
    solver = generate(),
    scorer = model_graded_qa(
      scorer_chat = scorer_chat,
      partial_credit = TRUE
    ),
    epochs = 3, # Run 3 evaluation rounds
    name = "An R Eval"
  )
}

#' Evaluate a model on the ARE dataset
#'
#' @param model OpenRouter model slug (e.g., "anthropic/claude-sonnet-4.6")
#' @param filename Output filename (without .rds extension). Defaults to model name.
#' @param scorer_chat Chat object used for model-graded scoring
#' @param overwrite Whether to overwrite existing results. Defaults to TRUE.
#' @param api_args List forwarded to chat_openrouter(api_args = ...).
#'   Should include `usage = list(include = TRUE)` to get authoritative
#'   per-request cost from OpenRouter.
#'
#' @return Invisible NULL. Results saved to results_rds/{filename}.rds
model_eval <- function(
  model,
  filename = model,
  scorer_chat,
  overwrite = TRUE,
  api_args = list(usage = list(include = TRUE))
) {
  model_path <- fs::path(results_dir, filename, ext = "rds")

  if (!overwrite & fs::file_exists(model_path)) {
    message(glue::glue("Skipping {model}: file already exists at {model_path}"))
    return(invisible(NULL))
  }

  solver_chat <- chat_openrouter(model = model, api_args = api_args)

  are_task <- create_are_task(scorer_chat)
  are_task$eval(solver_chat = solver_chat)

  readr::write_rds(are_task, file = model_path)
}
