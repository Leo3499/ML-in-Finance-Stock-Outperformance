# ============================================================
# Project: Will Stocks Outperform the S&P 500 next year ?
# ============================================================

# ------------------------------------------------------------
# Set working directory and data path
# ------------------------------------------------------------

# Automatically use the folder where this script is located
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

# ============================================================
# 1. Setup and Data Preparation
# ============================================================
library(randomForest)
library(readr)
library(caret)
library(pROC)
library(dplyr)
library(doParallel)
library(rpart)
library(rpart.plot)
library(nnet)
library(ggplot2)
library(ggcorrplot)

set.seed(600)

df <- read.csv("project_dataset.csv", stringsAsFactors = FALSE)

df$y_outperf_next <- factor(
  ifelse(grepl("^(true|yes|1)$", tolower(as.character(df$y_outperf_next))), "Yes", "No"),
  levels = c("No", "Yes")
)

# Define formula and predictors
id_vars <- intersect(c("ticker", "year"), names(df))
X_vars  <- setdiff(names(df), c("y_outperf_next", id_vars))
p <- length(X_vars)
form <- as.formula(paste("y_outperf_next ~", paste(X_vars, collapse = " + ")))

# ============================================================
# 2. Random Forest Model
# ============================================================

# ------------------------------------------------------------
# 2.1 Default Random Forest
# ------------------------------------------------------------
rf_default <- randomForest(
  formula = form, data = df,
  importance = TRUE, keep.forest = TRUE,
  keep.inbag = TRUE, ntree = 500
)
print(rf_default)

# Plot OOB error curve
plot(rf_default$err.rate[,"OOB"], type = "l",
     xlab = "Trees", ylab = "OOB error",
     main = "RF OOB error (classification)")

# Permutation importance
varImpPlot(rf_default, sort = TRUE, type = 1,
           main = "RF Variable Importance (Permutation)")

# ------------------------------------------------------------
# 2.2 Hyperparameter Tuning: 10-fold CV
# ------------------------------------------------------------
ctrl_cv10 <- trainControl(
  method = "cv", number = 10,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Grid of candidate mtry values (based on number of predictors)
mtry_grid <- sort(unique(pmax(1, round(c(sqrt(p)-2, sqrt(p), sqrt(p)+2, p/4, p/3)))))
ntree_values <- c(300, 500, 1000, 1500)

tune_results <- list()
for (nt in ntree_values) {
  set.seed(600)
  model_cv <- train(
    x = df[, X_vars],
    y = df$y_outperf_next,
    method = "rf",
    metric = "ROC",
    trControl = ctrl_cv10,
    tuneGrid = expand.grid(mtry = mtry_grid),
    ntree = nt
  )
  tune_results[[as.character(nt)]] <- model_cv$results
  
  print(model_cv)
  plot(model_cv, main = paste("10-fold CV: ROC vs mtry (ntree =", nt, ")"))
}

# Best model selection
collect   <- bind_rows(lapply(names(tune_results), function(k) mutate(tune_results[[k]], ntree = as.integer(k))))
best_row  <- arrange(collect, desc(ROC)) %>% slice(1)
best_ntree <- best_row$ntree
best_mtry  <- best_row$mtry
cat(sprintf("\nBest (10-fold): ntree=%d, mtry=%d, ROC=%.3f\n", best_ntree, best_mtry, best_row$ROC))

# ------------------------------------------------------------
# 2.3 Hyperparameter Tuning: Leave-One-Out CV
# ------------------------------------------------------------
ctrl_loocv <- trainControl(method = "LOOCV", classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           savePredictions = "final")

mtry_grid_loocv <- sort(unique(pmax(1, c(best_mtry - 1, best_mtry, best_mtry + 1))))
mtry_grid_loocv <- mtry_grid_loocv[mtry_grid_loocv <= p]

# Create parallel backend for LOOCV (much slower otherwise)
n_cores <- parallel::detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

ntree_loocv <- 3000

set.seed(600)
rf_loocv <- train(
  x = df[, X_vars],
  y = df$y_outperf_next,
  method = "rf",
  metric = "ROC",
  trControl = ctrl_loocv,
  tuneGrid = expand.grid(mtry = mtry_grid_loocv),
  ntree = ntree_loocv
)
print(rf_loocv)
plot(rf_loocv, main = paste("LOOCV: ROC vs mtry (ntree =", ntree_loocv, ")"))

best_mtry_loocv <- rf_loocv$bestTune$mtry
cat(sprintf("Best (LOOCV): ntree=%d, mtry=%d, ROC=%.3f\n",
            ntree_loocv, best_mtry_loocv, max(rf_loocv$results$ROC, na.rm = TRUE)))

stopCluster(cl)
registerDoSEQ()

# ------------------------------------------------------------
# 2.4 Final Random Forest Model
# ------------------------------------------------------------
ntree_final <- best_ntree
mtry_final  <- best_mtry_loocv

rf_final <- randomForest(
  formula = form, data = df,
  importance = TRUE, keep.inbag = TRUE,
  ntree = ntree_final, mtry = mtry_final
)
print(rf_final)

# OOB error rate
# rf_final has only 1000 trees, so we use rf_loocv (3000 trees)
# to visualize longer-run convergence and compute improvement.

oob_values <- rf_loocv$finalModel$err.rate[,"OOB"]

plot(oob_values, type = "l", lwd = 1, col = "black",
     xlab = "ntree", ylab = "OOB error rate",
     main = "Random Forest OOB error rate (3000 trees)", cex.main = 0.9,
     xaxs = "i", yaxs = "i")

relative_oob_improvement <- (oob_values[1000] - oob_values[2000]) / oob_values[1000]
cat(sprintf("\nRelative OOB improvement (1000→2000 trees): %.4f (%.2f%%)\n",
            relative_oob_improvement, relative_oob_improvement * 100))

# OOB AUC
pred_prob_oob <- rf_final$votes[,"Yes"]
pred_cls_oob  <- rf_final$predicted
auc_oob <- auc(roc(df$y_outperf_next, pred_prob_oob, levels = c("No", "Yes")))
cat(sprintf("OOB AUC = %.3f\n", as.numeric(auc_oob)))

# Feature importance
imp_mat <- importance(rf_final, type = 1, scale = TRUE)
imp_df  <- data.frame(Variable = rownames(imp_mat), MeanDecreaseAccuracy = imp_mat[,1]) %>%
  arrange(desc(MeanDecreaseAccuracy))
varImpPlot(rf_final, sort = TRUE, type = 1,
           main = "RF Variable Importance (Permutation / MeanDecreaseAccuracy)")

# ============================================================
# 3. Decision Tree Model
# ============================================================

# ------------------------------------------------------------
# 3.1 Baseline CART Model
# ------------------------------------------------------------
set.seed(600)
tree_model <- rpart(form, data = df, method = "class",
                    control = rpart.control(cp = 0.01))
print(tree_model)
rpart.plot(tree_model, main = "Decision Tree (CART)")

# Evaluate AUC and confusion matrix
tree_prob <- predict(tree_model, newdata = df, type = "prob")[,"Yes"]
tree_pred <- predict(tree_model, newdata = df, type = "class")
tree_auc  <- auc(roc(df$y_outperf_next, tree_prob, levels = c("No", "Yes")))
cat(sprintf("Decision Tree AUC = %.3f\n", as.numeric(tree_auc)))
confusionMatrix(tree_pred, df$y_outperf_next)

# Note: This model is trained and evaluated on the same data (in-sample).
# Its accuracy and AUC (≈0.83 / 0.86) are therefore optimistic.
# The cross-validated results in section 3.2 provide the fair, out-of-sample performance.

# ------------------------------------------------------------
# 3.2 Cross-Validated Decision Tree
# ------------------------------------------------------------
ctrl_cv10 <- trainControl(method = "cv", number = 10,
                          classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = "final")
tree_cv <- train(
  x = df[, X_vars], y = df$y_outperf_next,
  method = "rpart", metric = "ROC",
  trControl = ctrl_cv10, tuneLength = 10
)
print(tree_cv)
plot(tree_cv, main = "Decision Tree (10-fold CV)")

# Extract tree performance metrics
tree_pred_best <- tree_cv$pred
if (nrow(tree_pred_best) > 0) {
  tree_pred_best <- tree_pred_best[abs(tree_pred_best$cp - tree_cv$bestTune$cp) < 1e-8, ]
  tree_acc <- mean(tree_pred_best$pred == tree_pred_best$obs)
  tree_auc <- as.numeric(roc(tree_pred_best$obs, tree_pred_best$Yes, levels = c("No","Yes"))$auc)
} else {
  tree_acc <- tree_cv$results$Accuracy[1]
  tree_auc <- tree_cv$results$ROC[1]
}

# ------------------------------------------------------------
# 3.3 Pruned Decision Tree Visualization
# ------------------------------------------------------------
# Use the existing tree_model created above (section 3.1)

# Prune using 1-SE rule
cp_best <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]
cp_1se  <- with(as.data.frame(tree_model$cptable), {
  thr <- xerror[which.min(xerror)] + xstd[which.min(xerror)]
  CP[which(xerror <= thr)[1]]
})
tree_small <- prune(tree_model, cp = cp_1se)

# rename variables for readability in tree plot
nice_names <- c(beta_12m = "Market Beta (12m)", skew_12m = "Skewness (12m)",
                kurt_12m = "Kurtosis (12m)", vol_12m = "Volatility (12m)",
                avg_vol_3m = "Avg Volume (3m)", turnover_proxy_3m = "Turnover (3m)",
                max_dd_12m = "Max Drawdown (12m)", mom_6m = "Momentum (6m)",
                mom_12m = "Momentum (12m)", PE = "P/E", EVEBITDA = "EV/EBITDA",
                DivYield = "Dividend Yield", price_level_log = "Log Price",
                idio_vol_12m = "Idiosyncratic Vol (12m)", ret_y = "Return (12m)",
                ret_bmk_y = "Benchmark Return (12m)")

friendly <- function(x) ifelse(x %in% names(nice_names), nice_names[[x]], x)
split.fun <- function(x, labs, digits, varlen, faclen) {  # replaces default labels with clean names and thresholds
  sapply(labs, function(lbl) {
    parts <- strsplit(lbl, " ")[[1]]
    if (length(parts) >= 3) {
      var <- parts[1]; op <- parts[2]; thr <- as.numeric(parts[3])
      sprintf("%s %s %s", friendly(var), op, formatC(thr, format="f", digits=2))
    } else lbl
  })
}

rpart.plot(tree_small,
           main = "Decision Tree (Pruned) — Probability of Outperformance",
           type = 2, extra = 104, under = TRUE,
           box.palette = "Blues", fallen.leaves = TRUE,
           tweak = 1.2, branch.lwd = 2, shadow.col = "gray90",
           split.fun = split.fun)

# ============================================================
# 4. Neural Network Model
# ============================================================
set.seed(600)
ctrl_cv10_nn <- trainControl(
  method = "cv", number = 10,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Grid search over hidden-layer size ('size') and L2 regularization ('decay')
# 'preProcess = range' scales all predictors to [0,1] for neural nets
nn_grid <- expand.grid(size = c(3, 5, 7, 9),
                       decay = c(0, 0.001, 0.01, 0.1))

nn_cv <- train(
  x = df[, X_vars], y = df$y_outperf_next,
  method = "nnet", metric = "ROC",
  trControl = ctrl_cv10_nn, tuneGrid = nn_grid,
  preProcess = c("range"), maxit = 300, trace = FALSE
)
print(nn_cv)
plot(nn_cv, main = "Neural Net (10-fold CV)")

# NN performance
nn_best <- nn_cv$bestTune
nn_pred <- nn_cv$pred
nn_pred_best <- nn_pred[nn_pred$size == nn_best$size & nn_pred$decay == nn_best$decay, ]
nn_acc <- mean(nn_pred_best$pred == nn_pred_best$obs)
nn_auc <- as.numeric(roc(nn_pred_best$obs, nn_pred_best$Yes, levels = c("No","Yes"))$auc)
cat(sprintf("NN (CV): size=%d, decay=%.4f | Accuracy=%.3f, AUC=%.3f\n",
            nn_best$size, nn_best$decay, nn_acc, nn_auc))

# ============================================================
# 5. Model Comparison
# ============================================================
# Random Forest vs Neural Network (and Decision Tree) comparison
ctrl_cv10_rf <- trainControl(method = "cv", number = 10,
                             classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = "final")
rf_cv <- train(
  x = df[, X_vars], y = df$y_outperf_next,
  method = "rf", metric = "ROC",
  trControl = ctrl_cv10_rf,
  tuneGrid = data.frame(mtry = mtry_final),
  ntree = ntree_final
)
print(rf_cv)

rf_pred <- rf_cv$pred
rf_acc <- mean(rf_pred$pred == rf_pred$obs)
rf_auc <- as.numeric(roc(rf_pred$obs, rf_pred$Yes, levels = c("No","Yes"))$auc)
cat(sprintf("RF (CV): ntree=%d, mtry=%d | Accuracy=%.3f, AUC=%.3f\n",
            ntree_final, mtry_final, rf_acc, rf_auc))

compare_models <- data.frame(
  Model = c("Random Forest (10-fold CV)", "Neural Net (10-fold CV)", "Decision Tree (10-fold CV)"),
  Accuracy = c(rf_acc, nn_acc, tree_acc),
  AUC = c(rf_auc, nn_auc, tree_auc)
)
print(compare_models)

# ============================================================
# 6. Train/Test Evaluation
# ============================================================
set.seed(600)

# Create 80/20 holdout split to test final RF model on unseen data
train_index <- caret::createDataPartition(df$y_outperf_next, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data  <- df[-train_index, ]

rf_final_train <- randomForest(form, data = train_data,
                               ntree = ntree_final, mtry = mtry_final,
                               importance = TRUE)
rf_test_prob <- predict(rf_final_train, newdata = test_data, type = "prob")[,"Yes"]
rf_test_pred <- predict(rf_final_train, newdata = test_data, type = "class")

# Report holdout sizes and metrics
cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")
cat("Test Accuracy:", mean(rf_test_pred == test_data$y_outperf_next), "\n")
cat("Test AUC:", as.numeric(pROC::auc(
  pROC::roc(test_data$y_outperf_next, rf_test_prob, levels = c("No","Yes"))
)), "\n")

# ============================================================
# 7. Regime Performance Comparison
# ============================================================

# Evaluate pre-2020 vs post-2020 performance regimes
if ("year" %in% names(df)) {
  df_pre <- dplyr::filter(df, year < 2020)
  df_post <- dplyr::filter(df, year >= 2020)
  ok <- function(d) nrow(d) >= 50 && length(unique(d$y_outperf_next)) == 2
  
  if (ok(df_pre) && ok(df_post)) {
    set.seed(600)
    ctrl_regime <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                                summaryFunction = twoClassSummary, savePredictions = "final")
    
    fit_cv <- function(d) train(x = d[, X_vars], y = d$y_outperf_next, method = "rf",
                                metric = "ROC", trControl = ctrl_regime,
                                tuneGrid = data.frame(mtry = mtry_final), ntree = ntree_final)
    
    rf_pre_cv <- fit_cv(df_pre)
    rf_post_cv <- fit_cv(df_post)
    
    # Extract best predictions
    pick_best <- function(m) {
      p <- as.data.frame(m$pred)
      bt <- m$bestTune
      for (nm in names(bt)) if (nm %in% names(p)) {
        p <- if (is.numeric(bt[[nm]])) p[abs(p[[nm]] - bt[[nm]]) < 1e-8, , drop=FALSE]
        else p[as.character(p[[nm]]) == as.character(bt[[nm]]), , drop=FALSE]
      }
      if ("rowIndex" %in% names(p)) p <- p[!duplicated(p$rowIndex), , drop=FALSE]
      p
    }
    pre_pred_best <- pick_best(rf_pre_cv)
    post_pred_best <- pick_best(rf_post_cv)
    
    # Calculate metrics
    met <- function(p) {
      cm <- confusionMatrix(p$pred, p$obs, positive = "Yes")
      acc <- unname(cm$overall["Accuracy"])
      rec <- unname(cm$byClass["Sensitivity"])
      prec_name <- if ("Precision" %in% names(cm$byClass)) "Precision" else "Pos Pred Value"
      pre <- unname(cm$byClass[prec_name])
      c(Accuracy = acc, Recall = rec, Precision = pre)
    }
    
    m_pre <- met(pre_pred_best)
    m_post <- met(post_pred_best)
    
    regime_comparison <- data.frame(
      Regime = c("Pre-2020", "2020+"),
      Accuracy = round(c(m_pre["Accuracy"], m_post["Accuracy"]), 2),
      Recall = round(c(m_pre["Recall"], m_post["Recall"]), 2),
      Precision = round(c(m_pre["Precision"], m_post["Precision"]), 2)
    )
    
    cat("\n=== Model Performance Comparison by Regime ===\n")
    print(regime_comparison)
    readr::write_csv(regime_comparison, "data/rf_regime_performance_comparison.csv")
    cat("Saved: data/rf_regime_performance_comparison.csv\n")
  }
}

# ============================================================
# 8. Plots & ROC Comparisons
# ============================================================

# ------------------------------------------------------------
# 8.1 CV AUC Boxplot
# ------------------------------------------------------------
dir.create("data", showWarnings = FALSE)

df_cv_box <- list()
if (!is.null(rf_cv$resample) && "ROC" %in% names(rf_cv$resample))
  df_cv_box[["Random Forest"]] <- transmute(rf_cv$resample, Model="Random Forest", Fold=Resample, AUC=ROC)
if (!is.null(nn_cv$resample) && "ROC" %in% names(nn_cv$resample))
  df_cv_box[["Neural Network"]] <- transmute(nn_cv$resample, Model="Neural Network", Fold=Resample, AUC=ROC)
if (!is.null(tree_cv$resample) && "ROC" %in% names(tree_cv$resample))
  df_cv_box[["Decision Tree"]] <- transmute(tree_cv$resample, Model="Decision Tree", Fold=Resample, AUC=ROC)

cv_auc_df <- bind_rows(df_cv_box)
if (nrow(cv_auc_df)) {
  p_cv <- ggplot(cv_auc_df, aes(Model, AUC)) +
    geom_boxplot(outlier.alpha = .4, width = .6) +
    geom_jitter(width = .15, alpha = .5, size = 1.8) +
    labs(x = NULL, y = "AUC") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(face = "bold"))
  ggsave("data/cv_auc_boxplot.png", p_cv, width = 7, height = 4.5, dpi = 300)
  cat("Saved: data/cv_auc_boxplot.png\n")
}

# ------------------------------------------------------------
# 8.2 RF Feature Importance Plot
# ------------------------------------------------------------
if (exists("imp_df")) {
  imp_top10 <- imp_df %>%
    arrange(desc(MeanDecreaseAccuracy)) %>%
    slice_head(n = 10) %>%
    arrange(MeanDecreaseAccuracy) %>%
    mutate(Variable = factor(Variable, levels = Variable))
  p_imp <- ggplot(imp_top10, aes(MeanDecreaseAccuracy, Variable)) +
    geom_col() +
    labs(x = "Mean Decrease in Accuracy", y = NULL) +
    theme_minimal(base_size = 12)
  ggsave("data/rf_feature_importance.png", p_imp, width = 7, height = 4.5, dpi = 300)
  cat("Saved: data/rf_feature_importance.png\n")
}

# ------------------------------------------------------------
# 8.3 ROC by Regime (10-fold CV)
# ------------------------------------------------------------
if ("year" %in% names(df)) {
  df_pre  <- filter(df, year < 2020)
  df_post <- filter(df, year >= 2020)
  ok <- function(d) nrow(d) >= 50 && length(unique(d$y_outperf_next)) == 2 # check that subset is large enough and has both classes

  
  if (ok(df_pre) && ok(df_post)) {
    set.seed(600)
    ctrl_regime <- trainControl(
      method = "cv", number = 10,
      classProbs = TRUE, summaryFunction = twoClassSummary,
      savePredictions = "final"
    )
    
    fit_cv <- function(d) train(  # fit Random Forest with cross-validation on a given subset (used for pre-/post-2020)
      x = d[, X_vars], y = d$y_outperf_next,
      method = "rf", metric = "ROC",
      trControl = ctrl_regime,
      tuneGrid = data.frame(mtry = mtry_final),
      ntree = ntree_final
    )
    
    rf_pre_cv  <- fit_cv(df_pre)
    rf_post_cv <- fit_cv(df_post)
    
    pick_best <- function(m) {  # filter predictions corresponding to the best tuning parameters
      p <- as.data.frame(m$pred)
      bt <- m$bestTune
      for (nm in names(bt)) if (nm %in% names(p)) {
        p <- if (is.numeric(bt[[nm]]))
          p[abs(p[[nm]] - bt[[nm]]) < 1e-8, , drop = FALSE]
        else
          p[as.character(p[[nm]]) == as.character(bt[[nm]]), , drop = FALSE]
      }
      if ("rowIndex" %in% names(p)) p <- p[!duplicated(p$rowIndex), , drop = FALSE]
      p
    }
    
    pre_pred_best  <- pick_best(rf_pre_cv)
    post_pred_best <- pick_best(rf_post_cv)
    
    met <- function(p) {  # compute Accuracy, Recall, and Precision from predictions
      cm <- confusionMatrix(p$pred, p$obs, positive = "Yes")
      acc <- unname(cm$overall["Accuracy"])
      rec <- unname(cm$byClass["Sensitivity"])
      prec_name <- if ("Precision" %in% names(cm$byClass)) "Precision" else "Pos Pred Value"
      pre <- unname(cm$byClass[prec_name])
      c(Accuracy = acc, Recall = rec, Precision = pre)
    }
    
    m_pre  <- met(pre_pred_best)
    m_post <- met(post_pred_best)
    
    regime_comparison <- data.frame(
      Regime = c("Pre-2020", "2020+"),
      Accuracy = round(c(m_pre["Accuracy"],  m_post["Accuracy"]),  2),
      Recall   = round(c(m_pre["Recall"],    m_post["Recall"]),    2),
      Precision= round(c(m_pre["Precision"], m_post["Precision"]), 2)
    )
    
    cat("\n=== Model Performance Comparison by Regime (10-fold CV) ===\n")
    print(regime_comparison)
    readr::write_csv(regime_comparison, "data/rf_regime_performance_comparison.csv")
    cat("Saved: data/rf_regime_performance_comparison.csv\n")
  }
}

# ------------------------------------------------------------
# 8.4 Combined ROC (RF vs NN vs DT)
# ------------------------------------------------------------
get_oof <- function(model, pos = "Yes") {  # extract out-of-fold predicted probabilities from CV results
  pr <- as.data.frame(model$pred)
  if (!nrow(pr)) return(NULL)
  bt <- model$bestTune
  for (nm in names(bt)) if (nm %in% names(pr))
    pr <- if (is.numeric(bt[[nm]]))
      pr[abs(pr[[nm]] - bt[[nm]]) < 1e-8, , drop = FALSE]
  else
    pr[as.character(pr[[nm]]) == as.character(bt[[nm]]), , drop = FALSE]
  if (!nrow(pr)) return(NULL)
  if ("rowIndex" %in% names(pr)) pr <- pr[!duplicated(pr$rowIndex), , drop = FALSE]
  classes <- tryCatch(levels(model$trainingData$.outcome), error = function(e) c("No", "Yes"))
  prob_col <- if (pos %in% names(pr)) pos else intersect(classes, names(pr))[1]
  out <- pr[, c("obs", prob_col), drop = FALSE]
  names(out) <- c("obs", "prob")
  out <- out[complete.cases(out), , drop = FALSE]
  if (!nrow(out)) return(NULL)
  out
}

oof_rf   <- get_oof(rf_cv, "Yes")
oof_nn   <- get_oof(nn_cv, "Yes")
oof_tree <- get_oof(tree_cv, "Yes")

# Manual OOF for tree if needed
if (is.null(oof_tree) || !nrow(oof_tree)) {
  set.seed(600)
  folds <- createFolds(df$y_outperf_next, k = 10, list = TRUE)
  oof_tree <- data.frame(obs = df$y_outperf_next, prob = NA_real_)
  cp_use <- if (exists("cp_1se")) cp_1se else if (!is.null(tree_cv$bestTune$cp)) tree_cv$bestTune$cp else 0.01
  for (i in seq_along(folds)) {
    te <- folds[[i]]
    tr <- setdiff(seq_len(nrow(df)), te)
    fit <- rpart(form, data = df[tr,], method = "class", control = rpart.control(cp = cp_use))
    oof_tree$prob[te] <- predict(fit, df[te,], type = "prob")[,"Yes"]
  }
  oof_tree <- oof_tree[complete.cases(oof_tree), ]
}

# Calculate OOF AUCs
rf_auc_oof <- as.numeric(pROC::auc(pROC::roc(oof_rf$obs, oof_rf$prob, levels = c("No","Yes"), quiet = TRUE)))
nn_auc_oof <- as.numeric(pROC::auc(pROC::roc(oof_nn$obs, oof_nn$prob, levels = c("No","Yes"), quiet = TRUE)))
dt_auc_oof <- as.numeric(pROC::auc(pROC::roc(oof_tree$obs, oof_tree$prob, levels = c("No","Yes"), quiet = TRUE)))

cat(sprintf("OOF AUCs — RF: %.3f | NN: %.3f | DT: %.3f\n", rf_auc_oof, nn_auc_oof, dt_auc_oof))

# Save detailed comparison
compare_models_detailed <- data.frame(
  Model = c("Random Forest (10-fold CV)", "Neural Net (10-fold CV)", "Pruned Decision Tree (10-fold CV)"),
  Accuracy = c(rf_acc, nn_acc, tree_acc),
  AUC_CV = c(rf_auc, nn_auc, tree_auc),
  AUC_OOF = c(rf_auc_oof, nn_auc_oof, dt_auc_oof)
)
readr::write_csv(compare_models_detailed, "data/model_cv_comparison_detailed.csv")
cat("Saved: data/model_cv_comparison_detailed.csv\n")

# Combined ROC plot
mk_roc_df <- function(dat, label) {  # create ROC curve dataframe for plotting
  r <- pROC::roc(dat$obs, dat$prob, levels = c("No","Yes"), quiet = TRUE)
  data.frame(fpr = 1 - r$specificities, tpr = r$sensitivities,
             Model = sprintf("%s (AUC = %.3f)", label, as.numeric(pROC::auc(r))))
}
df_roc <- bind_rows(
  mk_roc_df(oof_rf, "Random Forest"),
  mk_roc_df(oof_nn, "Neural Net"),
  mk_roc_df(oof_tree, "Decision Tree")
)
p_all <- ggplot(df_roc, aes(fpr, tpr, color = Model)) +
  geom_line(linewidth = .9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = .6) +
  coord_equal() +
  labs(x = "1 - Specificity", y = "Sensitivity") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")
print(p_all)
ggsave("data/roc_rf_nn_tree.png", p_all, width = 7.2, height = 4.6, dpi = 300)
cat("Successfully saved: data/roc_rf_nn_tree.png\n")

# ============================================================
# 9. Statistical Analysis
# ============================================================

# Variation analysis of market vs fundamental features
market_vars <- c("vol_12m", "beta_12m", "skew_12m", "kurt_12m",
                 "mom_6m", "mom_12m", "max_dd_12m",
                 "avg_vol_3m", "turnover_proxy_3m")
fundamental_vars <- c("PE", "EVEBITDA", "Revenue", "NetIncome", "EBITDA", "DivYield")

variation_by_year <- df %>%
  group_by(year) %>%
  summarise(across(all_of(c(market_vars, fundamental_vars)), ~{
    mean_val <- mean(.x, na.rm = TRUE)
    if (!is.finite(mean_val) || mean_val <= 0) return(NA_real_)
    sd(.x, na.rm = TRUE) / mean_val
  }))

variation_mean <- summarise_all(variation_by_year, mean, na.rm = TRUE)
variation_summary <- data.frame(
  Variable = names(variation_mean),
  Mean_CV  = as.numeric(variation_mean[1, ])
) %>%
  mutate(Type = ifelse(Variable %in% market_vars, "Market-based", "Fundamental")) %>%
  filter(Mean_CV < 10)

variation_comparison_clean <- variation_summary %>%
  group_by(Type) %>%
  summarise(Average_CV = mean(Mean_CV, na.rm = TRUE))

print(variation_comparison_clean)

ggplot(variation_summary, aes(x = Type, y = Mean_CV)) +
  geom_boxplot(fill = "white", color = "black", width = 0.5, outlier.shape = NA) +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.1)) +
  labs(
    title = "Distribution of cross-sectional coefficients of variation by variable type",
    subtitle = "Extreme outliers (CV > 10) omitted for visual clarity",
    x = "", y = "Coefficient of Variation (CV)"
  ) +
  theme_minimal(base_size = 12)

# ============================================================
# 10. Save Artifacts
# ============================================================
dir.create("data", showWarnings = FALSE)

# Random Forest
write_csv(data.frame(ticker = df$ticker, year = df$year,
                     y_true = df$y_outperf_next,
                     prob_yes_oob = pred_prob_oob,
                     pred_oob = pred_cls_oob),
          "data/rf_final_oob_predictions.csv")
write_csv(collect, "data/rf_cv10_grid_results.csv")
write_csv(imp_df, "data/rf_final_permutation_importance.csv")
saveRDS(rf_default, "data/rf_default.rds")
saveRDS(rf_loocv, "data/rf_loocv_object.rds")
saveRDS(rf_final, "data/rf_final_model.rds")

# Decision Tree
write_csv(data.frame(ticker = df$ticker, year = df$year,
                     y_true = df$y_outperf_next,
                     prob_yes = tree_prob,
                     pred_tree = tree_pred),
          "data/tree_predictions.csv")
saveRDS(tree_model, "data/tree_model.rds")
saveRDS(tree_cv, "data/tree_cv_tuning.rds")

# Neural Network
saveRDS(nn_cv, "data/nn_cv_model.rds")
write_csv(nn_pred_best[, c("obs","pred","Yes","No","rowIndex","Resample")],
          "data/nn_cv_oof_predictions.csv")
write_csv(compare_models, "data/model_cv_comparison_rf_vs_nn.csv")

# Train/Test
write_csv(data.frame(ticker = test_data$ticker, year = test_data$year,
                     y_true = test_data$y_outperf_next,
                     prob_yes = rf_test_prob,
                     pred_test = rf_test_pred),
          "data/rf_test_predictions.csv")

cat("Saved all model artifacts successfully.\n")