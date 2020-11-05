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
knn2_rec <- 
  recipe(
    classification ~ enrl_grd + econ_dsvntg + tag_ed_fg + ayp_lep,
    data = full_train) %>%
  step_mutate(enrl_grd = factor(enrl_grd)) %>%
  step_unknown(all_nominal(), -all_outcomes()) %>%
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_dummy(enrl_grd, all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

#create a non-regular, space-filling design grid using grid_max_entropy and 25 parameter values
knn_params <- parameters(neighbors(range = c(1, 20)), dist_power())
knn_sfd <- grid_max_entropy(knn_params, size = 25)

#create ggplot
#knn_sfd %>%
#ggplot(aes(neighbors, dist_power)) +
#geom_point()

#create a KNN model object
knn2_mod <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification") %>%
  set_args(neighbors = tune(),
           dist_power = tune())

#fit tuned knn model and save as an object
knn2_res <- tune::tune_grid(
  knn2_mod,
  preprocessor = knn2_rec,
  resamples = train_cv,
  grid_knn_sfd,
  control = tune::control_resamples(save_pred = TRUE))

#save the fit resamples object as a .Rds file in the models folder using a relative path
saveRDS(knn2_res, "models/part4.Rds")

