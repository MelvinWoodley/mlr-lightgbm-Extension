

library(lightgbm)
library(mlr)

makeRLearner.regr.lightgbm = function() {
  makeRLearnerRegr(
    cl = "regr.lightgbm",
    package = "lightgbm",
    par.set = makeParamSet(
      
      makeIntegerLearnerParam(id ="num_iterations", default=100,lower=1),
      makeIntegerLearnerParam(id ="verbose",default=1),
      makeDiscreteLearnerParam(id = "boosting", default = "gbdt", values = c("gbdt", "dart","goss")), 
      makeNumericLearnerParam(id = "learning_rate", default = 0.1, lower = 0), 
      makeIntegerLearnerParam(id = "max_depth", default = -1, lower = -1),  
      makeIntegerLearnerParam(id = "min_data_in_leaf", default = 20, lower = 0), 
      makeIntegerLearnerParam(id = "num_leaves", default=31, lower=1),
      makeNumericLearnerParam(id = "feature_fraction", default = 1, lower = 0, upper = 1), 
      makeNumericLearnerParam(id = "bagging_fraction", default = 1, lower = 0, upper = 1),
      makeNumericLearnerParam(id = "bagging_freq", default = 0, lower = 0), 
      makeNumericLearnerParam(id = "min_gain_to_split", default = 0, lower = 0),
      makeLogicalLearnerParam(id="use_missing",default=TRUE,tunable = FALSE),
      makeNumericLearnerParam(id = "min_sum_hessian", default=10)
      
    ),
    par.vals = list(objective="regression"),
    properties = c("numerics", "weights","missings"),
    name = "LightGBM",
    short.name = "lightgbm",
    note = "First try at this"
  )
}


trainLearner.regr.lightgbm = function(.learner, .task, .subset, .weights = NULL,  ...) {
  f = getTaskDesc(.task)
  data = getTaskData(.task, .subset,target.extra = TRUE)
  lgb.data = lgb.Dataset(as.matrix(data$data), label = data$target)
  lightgbm::lgb.train(data = lgb.data,objective="regression",
                      verbosity = -1, 
                      verbose = -1,
                      record = TRUE,...)
  
}



predictLearner.regr.lightgbm= function(.learner, .model, .newdata, ...) {
  predict(.model$learner.model, as.matrix(.newdata))
}


registerS3method("makeRLearner", "regr.lightgbm", makeRLearner.regr.lightgbm)
registerS3method("trainLearner", "regr.lightgbm", trainLearner.regr.lightgbm)
registerS3method("predictLearner", "regr.lightgbm", predictLearner.regr.lightgbm)

