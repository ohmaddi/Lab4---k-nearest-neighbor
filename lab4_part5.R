#load tidyverse and tidymodels
library(tidyverse)
library(tidymodels)


#add code to prepare for parallel processing
all_cores <- parallel::detectCores(logical = FALSE)

library(doParallel)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})

#read in train.csv from data folder using a relative path
full_train <- read_csv("data/train.csv") %>%
  mutate(classification = factor(classification, 
                                 levels = 1:4,
                                 labels = c("far below", "below",
                                            "meets", "exceeds"),
                                 ordered = TRUE))

#randomly select a proportion of 0.005 of all rows (comment out when running on talapas)
set.seed(584)
#full_train <- slice_sample(full_train, prop = 0.005)
full_train <- slice_sample(full_train, prop = .10)

#create initial split object, pull training data from it, create k-fold cross validation object
split <- initial_split(full_train)
train <- training(split)
train_cv <- vfold_cv(train)

#create basic recipe to work with for the lab.
#use classification as outcome (not score) and do not model all variables.
knn3_rec <- 
  recipe(
    classification ~ enrl_grd + econ_dsvntg + tag_ed_fg + ayp_lep,
    data = full_train) %>%
  step_mutate(enrl_grd = factor(enrl_grd)) %>%
  step_unknown(all_nominal(), -all_outcomes()) %>%
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_dummy(enrl_grd, all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

#create a KNN model object
knn3_mod <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")

#read in tuned model from part 4
p4tuned <- readRDS("models/part4 (local).Rds")

#conduct final fit
#select best tuning parameters
knn_best <- p4tuned %>% 
  select_best(metric = "roc_auc")

#finalize model using best tuning parameters
knn_mod_final <- knn3_mod %>%
  finalize_model(knn_best)

#finalize recipe using best tuning parameters
knn_rec_final <- knn3_rec %>%
  finalize_recipe(knn_best)

#run final fit on initial data split: couldn't get it to run with parallel processing (re: email error, so moving forward without)
#c1 <- makeCluster(8)
#registerDoParallel(c1)
#knn_final_res <- last_fit(
  #knn_mod_final,
  #preprocessor = knn_rec_final,
  #split = split)
#stopCluster(c1)

knn_final_res <- last_fit(
  knn_mod_final,
  preprocessor = knn_rec_final,
  split = split)

#saveRDS
saveRDS(knn_final_res, "models/part5.Rds")





