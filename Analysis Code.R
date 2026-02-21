rm(list = ls()); gc()

# ==============================================================================
# ASASF 2026 - COMPETITION PIPELINE 
# ==============================================================================

set.seed(2026)

# ------------------------------------------------------------------------------
# Packages 
# ------------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(tidyr)
  library(ggplot2)
  library(recipes)
  library(parsnip)
  library(workflows)
  library(tune)
  library(rsample)
  library(yardstick)
  library(dials)
  library(stacks)
  library(doParallel)
  library(rlang)
  library(glmnet)
  library(xgboost)
  library(brulee)
  library(ranger)
  library(rules)
  library(Cubist)
  library(earth)
  library(kernlab)
  library(dbarts)
})

# ------------------------------------------------------------------------------
# Parallel backend (parallelize over resamples)
# ------------------------------------------------------------------------------
n_cores <- max(1L, parallel::detectCores(logical = TRUE) - 1L)
cl <- parallel::makePSOCKcluster(n_cores)
doParallel::registerDoParallel(cl)
on.exit({ try(parallel::stopCluster(cl), silent = TRUE) }, add = TRUE)
cat("Parallel workers:", n_cores, "\n")

# ------------------------------------------------------------------------------
# Data import
# ------------------------------------------------------------------------------
tmp <- tempfile()
download.file("https://luminwin.github.io/ASASF/train.rds", tmp, mode = "wb")
train_raw <- readRDS(tmp)
download.file("https://luminwin.github.io/ASASF/test.rds", tmp, mode = "wb")
test_raw <- readRDS(tmp)

y <- "LBDHDD_outcome"
stopifnot(y %in% names(train_raw))
stopifnot(!(y %in% names(test_raw)))

cat("Train:", paste(dim(train_raw), collapse = " x "),
    " Test:", paste(dim(test_raw),  collapse = " x "), "\n")
cat("Outcome summary:\n"); print(summary(train_raw[[y]]))

# ------------------------------------------------------------------------------
# Quick EDA
# ------------------------------------------------------------------------------
print(
  ggplot(train_raw, aes(x = .data[[y]])) +
    geom_histogram(bins = 40) +
    labs(title = "Outcome distribution", x = y, y = "Count")
)

miss_map <- sort(colMeans(is.na(train_raw)) * 100, decreasing = TRUE)
cat("\nTop missingness (%):\n"); print(head(miss_map, 12))

# ------------------------------------------------------------------------------
# Sentinel detection + cleaning
# ------------------------------------------------------------------------------
num_vars <- setdiff(names(train_raw)[vapply(train_raw, is.numeric, logical(1))], y)

sentinel_check <- map_dfr(num_vars, ~{
  x <- train_raw[[.x]]
  tibble(
    var = .x,
    max = suppressWarnings(max(x, na.rm = TRUE)),
    p99 = suppressWarnings(as.numeric(quantile(x, 0.99, na.rm = TRUE)))
  )
}) %>%
  filter(is.finite(max), is.finite(p99), p99 > 0, max > 5 * p99) %>%
  arrange(desc(max))

cat("\nColumns flagged for possible sentinel codes:\n")
print(head(sentinel_check, 20))
sentinel_cols <- sentinel_check$var

clean_sentinels <- function(df, cols,
                            sentinels = c(7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999)) {
  cols <- intersect(cols, names(df))
  cols <- cols[vapply(df[cols], is.numeric, logical(1))]
  if (length(cols) == 0) return(df)
  df %>% mutate(across(all_of(cols), ~ ifelse(.x %in% sentinels, NA_real_, .x)))
}

train_clean <- clean_sentinels(train_raw, sentinel_cols)
test_clean  <- clean_sentinels(test_raw,  sentinel_cols)

# Drop constant columns (never drop outcome)
const_cols <- names(train_clean)[vapply(train_clean, function(z) dplyr::n_distinct(z, na.rm = FALSE) <= 1, logical(1))]
const_cols <- setdiff(const_cols, y)

train_clean <- train_clean %>% select(-any_of(const_cols))
test_clean  <- test_clean  %>% select(any_of(setdiff(names(train_clean), y)))

# Align column order (outcome first)
train_clean <- train_clean %>% select(all_of(c(y, setdiff(names(train_clean), y))))

cat("\nAfter cleaning: Train:", paste(dim(train_clean), collapse = " x "),
    " Test:", paste(dim(test_clean),  collapse = " x "), "\n")

# ------------------------------------------------------------------------------
# Cap extreme numeric predictors using train-only quantile bounds
# ------------------------------------------------------------------------------
winsorize_from_train <- function(train_df, test_df, cols, p = 0.005) {
  cols <- intersect(cols, names(train_df))
  cols <- cols[vapply(train_df[cols], is.numeric, logical(1))]
  if (length(cols) == 0) return(list(train = train_df, test = test_df))
  
  bounds <- lapply(cols, function(v) {
    x <- train_df[[v]]
    if (all(is.na(x))) return(NULL)
    qs <- quantile(x, probs = c(p, 1 - p), na.rm = TRUE, names = FALSE, type = 7)
    if (!is.finite(qs[1]) || !is.finite(qs[2]) || qs[1] >= qs[2]) return(NULL)
    list(var = v, lo = qs[1], hi = qs[2])
  })
  bounds <- Filter(Negate(is.null), bounds)
  
  cap_df <- function(df) {
    for (b in bounds) {
      v <- b$var
      df[[v]] <- pmin(pmax(df[[v]], b$lo), b$hi)
    }
    df
  }
  
  list(train = cap_df(train_df), test = cap_df(test_df))
}

num_cols_now <- setdiff(names(train_clean)[vapply(train_clean, is.numeric, logical(1))], y)
w <- winsorize_from_train(train_clean, test_clean, num_cols_now, p = 0.005)
train_clean <- w$train
test_clean  <- w$test
rm(w); gc()

# ------------------------------------------------------------------------------
# Resampling + tuning control
# ------------------------------------------------------------------------------
folds  <- vfold_cv(train_clean, v = 5, repeats = 2, strata = !!sym(y))  # FIXED
metric <- metric_set(rmse)

ctrl_grid_stack <- control_grid(
  save_pred     = TRUE,
  save_workflow = TRUE,
  parallel_over = "resamples",
  verbose       = TRUE
)

# ------------------------------------------------------------------------------
# Recipes
# ------------------------------------------------------------------------------
rec_tree <- recipe(as.formula(paste0(y, " ~ .")), data = train_clean) %>%
  step_indicate_na(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

rec_norm <- recipe(as.formula(paste0(y, " ~ .")), data = train_clean) %>%
  step_indicate_na(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

rec_nn <- rec_norm %>%
  step_mutate(!!y := as.numeric(.data[[y]]))

# ------------------------------------------------------------------------------
# Helper: safe tuning wrapper
# ------------------------------------------------------------------------------
safe_tune <- function(name, wf, grid) {
  cat("\n--- Tuning:", name, "---\n")
  out <- tryCatch(
    tune_grid(
      wf,
      resamples = folds,
      grid      = grid,
      metrics   = metric,
      control   = ctrl_grid_stack
    ),
    error = function(e) {
      message("Model failed: ", name, " | ", conditionMessage(e))
      NULL
    }
  )
  out
}

# ------------------------------------------------------------------------------
# MODEL 1: Ridge
# ------------------------------------------------------------------------------
ridge_spec <- linear_reg(penalty = tune(), mixture = 0) %>% set_engine("glmnet")
ridge_grid <- grid_regular(penalty(range = c(-6, 2)), levels = 60)
ridge_wf   <- workflow() %>% add_recipe(rec_norm) %>% add_model(ridge_spec)
ridge_res  <- safe_tune("Ridge (glmnet)", ridge_wf, ridge_grid)
if (!is.null(ridge_res)) {
  cat("\nTop Ridge configs:\n")
  print(show_best(ridge_res, metric = "rmse", n = 5))
}

# ------------------------------------------------------------------------------
# MODEL 2: XGBoost
# ------------------------------------------------------------------------------
tmp_prep <- prep(rec_tree, training = train_clean %>% slice(1:min(300, nrow(train_clean))), verbose = FALSE)
p_post <- ncol(juice(tmp_prep)) - 1
rm(tmp_prep); gc()

mtry_upper <- max(10L, min(300L, as.integer(p_post)))
mtry_lower <- max(5L,  min(40L,  mtry_upper))

lambda_param <- dials::new_quant_param(
  type = "double", range = c(-4, 2), inclusive = c(TRUE, TRUE),
  trans = scales::log10_trans(), label = c(lambda = "xgboost lambda (L2)")
)
alpha_param <- dials::new_quant_param(
  type = "double", range = c(0, 1), inclusive = c(TRUE, TRUE),
  trans = scales::identity_trans(), label = c(alpha = "xgboost alpha (L1)")
)

xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  min_n          = tune(),
  sample_size    = tune(),
  mtry           = tune()
) %>%
  set_mode("regression") %>%
  set_engine(
    "xgboost",
    objective = "reg:squarederror",
    nthread   = 1,
    lambda    = tune(),
    alpha     = tune()
  )

xgb_grid <- grid_space_filling(
  trees(range = c(800L, 3500L)),
  tree_depth(range = c(2L, 8L)),
  learn_rate(range = c(-3.5, -1.0)),
  loss_reduction(range = c(-6, 1.0)),
  min_n(range = c(5L, 150L)),
  sample_size = sample_prop(range = c(0.6, 0.95)),
  mtry(range = c(mtry_lower, mtry_upper)),
  lambda_param,
  alpha_param,
  size = 80
)

xgb_wf  <- workflow() %>% add_recipe(rec_tree) %>% add_model(xgb_spec)
xgb_res <- safe_tune("XGBoost (xgboost)", xgb_wf, xgb_grid)
if (!is.null(xgb_res)) {
  cat("\nTop XGB (best mean/std_err):\n")
  print(show_best(xgb_res, metric = "rmse", n = 1) %>% dplyr::select(mean, std_err))
  cat("\nTop XGB configs:\n")
  print(show_best(xgb_res, metric = "rmse", n = 5))
}

# ------------------------------------------------------------------------------
# MODEL 3: Neural Net (brulee)
# ------------------------------------------------------------------------------
nn_spec <- mlp(
  hidden_units = tune(),
  dropout      = tune(),
  epochs       = tune(),
  learn_rate   = tune()
) %>%
  set_mode("regression") %>%
  set_engine("brulee")

nn_grid <- grid_latin_hypercube(
  hidden_units(range = c(32L, 512L)),
  dropout(range = c(0.0, 0.5)),
  epochs(range = c(80L, 350L)),
  learn_rate(range = c(-4, -1)),
  size = 35
)

nn_wf  <- workflow() %>% add_recipe(rec_nn) %>% add_model(nn_spec)
nn_res <- safe_tune("Neural Net (brulee)", nn_wf, nn_grid)
if (!is.null(nn_res)) {
  cat("\nTop NN configs:\n")
  print(show_best(nn_res, metric = "rmse", n = 5))
}

# ------------------------------------------------------------------------------
# MODEL 4: Random Forest (ranger)
# ------------------------------------------------------------------------------
rf_spec <- rand_forest(trees = tune(), mtry = tune(), min_n = tune()) %>%
  set_mode("regression") %>%
  set_engine("ranger", num.threads = 1, importance = "none")

rf_grid <- grid_latin_hypercube(
  trees(range = c(800L, 3000L)),
  min_n(range  = c(2L, 50L)),
  mtry(range   = c(mtry_lower, mtry_upper)),
  size = 40
)

rf_wf  <- workflow() %>% add_recipe(rec_tree) %>% add_model(rf_spec)
rf_res <- safe_tune("Random Forest (ranger)", rf_wf, rf_grid)
if (!is.null(rf_res)) {
  cat("\nTop RF configs:\n")
  print(show_best(rf_res, metric = "rmse", n = 5))
}

# ------------------------------------------------------------------------------
# MODEL 5: Cubist
# ------------------------------------------------------------------------------
cub_spec <- cubist_rules(committees = tune(), neighbors = tune()) %>%
  set_mode("regression") %>%
  set_engine("Cubist")

cub_grid <- grid_latin_hypercube(
  committees(range = c(10L, 120L)),
  neighbors(range  = c(0L, 9L)),
  size = 40
)

cub_wf  <- workflow() %>% add_recipe(rec_tree) %>% add_model(cub_spec)
cub_res <- safe_tune("Cubist (Cubist)", cub_wf, cub_grid)
if (!is.null(cub_res)) {
  cat("\nTop Cubist configs:\n")
  print(show_best(cub_res, metric = "rmse", n = 5))
}

# ------------------------------------------------------------------------------
# MODEL 6: MARS (earth)
# ------------------------------------------------------------------------------
mars_spec <- mars(num_terms = tune(), prod_degree = tune()) %>%
  set_mode("regression") %>%
  set_engine("earth")

mars_grid <- grid_latin_hypercube(
  num_terms(range = c(10L, 80L)),
  prod_degree(range = c(1L, 3L)),
  size = 35
)

mars_wf  <- workflow() %>% add_recipe(rec_norm) %>% add_model(mars_spec)
mars_res <- safe_tune("MARS (earth)", mars_wf, mars_grid)
if (!is.null(mars_res)) {
  cat("\nTop MARS configs:\n")
  print(show_best(mars_res, metric = "rmse", n = 5))
}

# ------------------------------------------------------------------------------
# MODEL 7: SVM RBF (kernlab)
# ------------------------------------------------------------------------------
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_mode("regression") %>%
  set_engine("kernlab")

svm_grid <- grid_latin_hypercube(
  cost(range = c(-5, 5)),
  rbf_sigma(range = c(-8, 0)),
  size = 35
)

svm_wf  <- workflow() %>% add_recipe(rec_norm) %>% add_model(svm_spec)
svm_res <- safe_tune("SVM RBF (kernlab)", svm_wf, svm_grid)
if (!is.null(svm_res)) {
  cat("\nTop SVM configs:\n")
  print(show_best(svm_res, metric = "rmse", n = 5))
}


# ------------------------------------------------------------------------------
# MODEL 8: BART 
# ------------------------------------------------------------------------------
bart_res <- NULL

if (exists("bart", where = asNamespace("parsnip"), mode = "function")) {
  
  bart_spec <- parsnip::bart(
    trees = tune()
  ) %>%
    set_mode("regression") %>%
    set_engine("dbarts")
  
  bart_grid <- tibble(trees = c(50L, 100L, 200L))
  
  bart_wf  <- workflow() %>% add_recipe(rec_tree) %>% add_model(bart_spec)
  bart_res <- safe_tune("BART (dbarts)", bart_wf, bart_grid)
  
  if (!is.null(bart_res)) {
    cat("\nTop BART configs:\n")
    print(show_best(bart_res, metric = "rmse", n = 5))
  }
  
} else {
  message("Skipping BART: parsnip::bart() not available (update parsnip).")
}


# ------------------------------------------------------------------------------
# STACKED ENSEMBLE
# ------------------------------------------------------------------------------
ens <- stacks()
for (obj in list(ridge_res, xgb_res, nn_res, rf_res, cub_res, mars_res, svm_res, bart_res)) {
  if (!is.null(obj)) ens <- ens %>% add_candidates(obj)
}

ens_fit <- ens %>%
  blend_predictions(metric = metric) %>%
  fit_members()

cat("\nEnsemble summary:\n")
print(ens_fit)

# ------------------------------------------------------------------------------
# Predict test
# ------------------------------------------------------------------------------
pred_final <- predict(ens_fit, new_data = test_clean) %>%
  dplyr::transmute(pred = .pred)

cat("\nPrediction summary:\n")
print(summary(pred_final$pred))

print(
  ggplot() +
    geom_density(data = train_clean, aes(x = .data[[y]], fill = "Train"), alpha = 0.3) +
    geom_density(data = pred_final,  aes(x = pred,       fill = "Test pred"), alpha = 0.3) +
    labs(title = "Train outcome vs test predictions", x = "Value", y = "Density")
)

# write to CSV
readr::write_csv(pred_final, "pred.csv")



















