# This file defines:
# - create_are_task(): Function to create the evaluation task
# - model_eval(): Core function to evaluate a single model

library(ellmer)
library(vitals)

# Set results directory
results_dir <- here::here("results_rds")

#' Custom robust scorer that handles API rate limits and errors
#'
#' @param template Prompt template
#' @param instructions Grading instructions
#' @param grade_pattern Regex to extract the grade
#' @param partial_credit Whether to allow partial credit
#' @param scorer_chat The ellmer chat object to use
#' @param max_active Maximum number of concurrent requests
#' @param rpm Requests per minute limit
robust_model_graded_qa <- function(
  template = NULL,
  instructions = NULL,
  grade_pattern = "(?i)GRADE\\s*:\\s*([CPI])(.*)$",
  partial_credit = FALSE,
  scorer_chat = NULL,
  max_active = 2,
  rpm = 20
) {
  ch <- scorer_chat
  function(samples, ..., scorer_chat = ch) {
    qa_default_template <- getFromNamespace("qa_default_template", "vitals")
    qa_default_instructions <- getFromNamespace("qa_default_instructions", "vitals")
    qa_format_prompt <- getFromNamespace("qa_format_prompt", "vitals")
    qa_extract_grade <- getFromNamespace("qa_extract_grade", "vitals")
    process_grades <- getFromNamespace("process_grades", "vitals")

    template <- template %||% qa_default_template()
    instructions <- instructions %||% qa_default_instructions(partial_credit)

    prompts <- purrr::map_chr(seq_len(nrow(samples)), function(i) {
      qa_format_prompt(
        template, samples$input[i], samples$result[i],
        samples$target[i], instructions
      )
    })

    if (is.null(scorer_chat)) {
      solver_chat <- getFromNamespace("solver_chat", "vitals")
      scorer_chat <- solver_chat(samples[1, ])
    }
    scorer_chat <- scorer_chat$clone()

    message("Scoring ", length(prompts), " responses...")
    responses <- ellmer::parallel_chat(
      scorer_chat,
      as.list(prompts),
      max_active = max_active,
      rpm = rpm
    )

    grades <- purrr::map_chr(responses, function(response_chat) {
      if (inherits(response_chat, "error")) {
        message("API error during scoring: ", conditionMessage(response_chat))
        return(NA_character_)
      }
      response_text <- response_chat$last_turn()@text
      qa_extract_grade(response_text, grade_pattern, partial_credit) %||% NA_character_
    })

    scores <- process_grades(grades, partial_credit)

    metadata <- purrr::map(seq_along(prompts), function(i) {
      if (inherits(responses[[i]], "error")) {
        list(
          prompt = prompts[i],
          response = NA_character_,
          error = conditionMessage(responses[[i]]),
          grade_pattern = grade_pattern
        )
      } else {
        list(
          prompt = prompts[i],
          response = responses[[i]]$last_turn()@text,
          grade_pattern = grade_pattern
        )
      }
    })

    list(score = scores, scorer_chat = responses, scorer_metadata = metadata)
  }
}

#' Create the ARE evaluation task
#'
#' @param scorer_chat Chat object used for model-graded scoring
#' @return A Task object configured for ARE evaluation
create_are_task <- function(scorer_chat) {
  Task$new(
    dataset = are,
    solver = generate(),
    scorer = robust_model_graded_qa(
      scorer_chat = scorer_chat,
      partial_credit = TRUE,
      max_active = 2,
      rpm = 20
    ),
    epochs = 3,
    name = "An R Eval"
  )
}

#' Evaluate a model on the ARE dataset
#'
#' @param model API model identifier (e.g., "anthropic/claude-sonnet-4-20250514")
#' @param filename Output filename (without .rds extension). Defaults to model name.
#' @param scorer_chat Chat object used for model-graded scoring
#' @param overwrite Whether to overwrite existing results. Defaults to TRUE.
#' @param ... Additional arguments passed to chat():
#'   - base_url: Custom API endpoint
#'   - api_key: Custom API key
#'   - api_args: List of additional API arguments (e.g., thinking config)
#'
#' @return Invisible NULL. Results saved to results_rds/{filename}.rds
model_eval <- function(
  model,
  filename = model,
  scorer_chat,
  overwrite = TRUE,
  ...
) {
  model_path <- fs::path(results_dir, filename, ext = "rds")

  if (!overwrite & fs::file_exists(model_path)) {
    message(glue::glue("Skipping {model}: file already exists at {model_path}"))
    return(invisible(NULL))
  }

  extra_args <- list(...)
  base_url <- extra_args$base_url

  if (!is.null(base_url) && grepl("openrouter", base_url, ignore.case = TRUE)) {
    chat <- ellmer::chat_openai_compatible(
      base_url = base_url,
      model = model,
      credentials = extra_args$credentials,
      api_args = extra_args$api_args %||% list()
    )
  } else {
    chat <- ellmer::chat(name = model, ...)
  }

  are_task <- create_are_task(scorer_chat)
  are_task$eval(solver_chat = chat)

  readr::write_rds(are_task, file = model_path)
}
