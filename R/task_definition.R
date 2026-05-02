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

#' Checkpointed solver that saves progress after every N samples
#'
#' Wraps generate() so that if solving fails partway through, we can resume
#' from the last checkpoint without re-processing already-solved samples.
#'
#' @param solver_chat ellmer Chat object (or NULL if passed at call time)
#' @param checkpoint_path Path to save/load checkpoint RDS
#' @param batch_size Number of samples to process before writing a checkpoint
#' @return A solver function compatible with vitals::Task
checkpointed_generate <- function(solver_chat = NULL, checkpoint_path = NULL, batch_size = 5) {
  chat <- solver_chat
  function(inputs, ..., solver_chat = chat) {
    if (is.function(solver_chat)) {
      ch <- solver_chat()
      check_inherits(ch, "Chat")
    } else {
      check_inherits(solver_chat, "Chat")
      ch <- solver_chat$clone()
    }

    total_inputs <- length(inputs)
    completed <- 0
    results_result <- character(0)
    results_chat <- list()

    # Load checkpoint if it exists
    if (!is.null(checkpoint_path) && fs::file_exists(checkpoint_path)) {
      checkpoint <- readr::read_rds(checkpoint_path)
      completed <- checkpoint$completed
      results_result <- checkpoint$result
      results_chat <- checkpoint$solver_chat
      message(
        "Resuming solve from checkpoint: ", completed,
        " of ", total_inputs, " already completed"
      )
      inputs <- inputs[(completed + 1):total_inputs]
    }

    n_remaining <- length(inputs)
    if (n_remaining == 0) {
      message("All samples already solved from checkpoint.")
      return(list(result = results_result, solver_chat = results_chat))
    }

    # Process remaining inputs in batches
    for (i in seq(1, n_remaining, by = batch_size)) {
      batch_end <- min(i + batch_size - 1, n_remaining)
      batch_inputs <- inputs[i:batch_end]
      global_start <- completed + i
      global_end <- completed + batch_end
      message(
        "Solving batch ", ceiling(i / batch_size),
        " (samples ", global_start, "-", global_end, " of ", total_inputs, ")"
      )

      res <- ellmer::parallel_chat(ch, as.list(batch_inputs), ...)

      results_result <- c(results_result, purrr::map_chr(res, function(c) c$last_turn()@text))
      results_chat <- c(results_chat, res)

      # Save checkpoint after every batch
      if (!is.null(checkpoint_path)) {
        readr::write_rds(
          list(
            completed = completed + batch_end,
            result = results_result,
            solver_chat = results_chat
          ),
          checkpoint_path
        )
      }
    }

    # Clean up checkpoint on full success
    if (!is.null(checkpoint_path) && fs::file_exists(checkpoint_path)) {
      fs::file_delete(checkpoint_path)
      message("Checkpoint removed after successful solve.")
    }

    list(result = results_result, solver_chat = results_chat)
  }
}

#' Create the ARE evaluation task
#'
#' @param scorer_chat Chat object used for model-graded scoring
#' @param solver Optional custom solver function. Defaults to generate().
#' @return A Task object configured for ARE evaluation
create_are_task <- function(scorer_chat, solver = generate()) {
  Task$new(
    dataset = are,
    solver = solver,
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
#' Saves checkpoints during solving so a failure mid-solve can be resumed.
#' Also saves intermediate results after solve so a scoring failure can be
#' resumed without re-solving.
#'
#' @param model API model identifier
#' @param filename Output filename (without .rds extension). Defaults to model name.
#' @param scorer_chat Chat object used for model-graded scoring
#' @param overwrite Whether to overwrite existing results. Defaults to TRUE.
#' @param resume If TRUE, skip solving and resume from scoring.
#' @param ... Additional arguments passed to chat()
#'
#' @return Invisible NULL. Results saved to results_rds/{filename}.rds
model_eval <- function(
  model,
  filename = model,
  scorer_chat,
  overwrite = TRUE,
  resume = FALSE,
  ...
) {
  model_path <- fs::path(results_dir, filename, ext = "rds")
  solved_path <- fs::path(results_dir, paste0(filename, "_solved"), ext = "rds")
  checkpoint_path <- fs::path(results_dir, paste0(filename, "_solve_checkpoint"), ext = "rds")

  if (!overwrite & fs::file_exists(model_path)) {
    message(glue::glue("Skipping {model}: final results already exist at {model_path}"))
    return(invisible(NULL))
  }

  # Clear stale checkpoints when starting fresh
  if (isTRUE(overwrite) && fs::file_exists(checkpoint_path)) {
    fs::file_delete(checkpoint_path)
    message("Removed stale checkpoint: ", checkpoint_path)
  }

  # ---------------------------------------------------------------------------
  # Resume from a saved solve file if available
  # ---------------------------------------------------------------------------
  if (!isTRUE(resume) && fs::file_exists(solved_path) && !fs::file_exists(model_path)) {
    message("Found intermediate solve results. Automatically resuming scoring.")
    resume <- TRUE
  }

  if (isTRUE(resume) && fs::file_exists(solved_path)) {
    message("Resuming from saved solve results: ", solved_path)
    are_task <- readr::read_rds(solved_path)
  } else {
    # -------------------------------------------------------------------------
    # Create solver chat
    # -------------------------------------------------------------------------
    extra_args <- list(...)
    base_url <- extra_args$base_url

    message("[1/4] Creating solver chat for model: ", model)
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
    message("[1/4] Solver chat created successfully.")

    # -------------------------------------------------------------------------
    # Create task with checkpointed solver and solve
    # -------------------------------------------------------------------------
    message("[2/4] Creating ARE task with checkpointed solver...")
    custom_solver <- checkpointed_generate(
      solver_chat = chat,
      checkpoint_path = checkpoint_path,
      batch_size = 5
    )
    are_task <- create_are_task(scorer_chat, solver = custom_solver)
    message("[2/4] ARE task created successfully.")

    message("[3/4] Solving (may take a while)...")
    tryCatch(
      {
        are_task$solve()
        message("[3/4] Solving completed. Saving intermediate results...")
        readr::write_rds(are_task, file = solved_path)
        message("Intermediate results saved to: ", solved_path)
      },
      error = function(e) {
        message(glue::glue("[3/4] Solving FAILED: {e$message}"))
        if (fs::file_exists(checkpoint_path)) {
          message(glue::glue(
            "Checkpoint preserved at {checkpoint_path}. Rerun to resume solving."
          ))
        }
        stop(e)
      }
    )
  }

  # ---------------------------------------------------------------------------
  # Score
  # ---------------------------------------------------------------------------
  message("[4/4] Scoring solved results...")
  tryCatch(
    {
      are_task$score()
      message("[4/4] Scoring completed.")

      are_task$measure()
      are_task$log(are_task$dir)
      readr::write_rds(are_task, file = model_path)

      # Clean up intermediate files on success
      if (fs::file_exists(solved_path)) {
        fs::file_delete(solved_path)
        message("Cleaned up intermediate file: ", solved_path)
      }
      if (fs::file_exists(checkpoint_path)) {
        fs::file_delete(checkpoint_path)
        message("Cleaned up checkpoint file: ", checkpoint_path)
      }
      message("Final results saved to: ", model_path)
    },
    error = function(e) {
      message(glue::glue("[4/4] Scoring FAILED: {e$message}"))
      message(glue::glue(
        "Intermediate solve results are preserved at {solved_path}"
      ))
      stop(e)
    }
  )
}
