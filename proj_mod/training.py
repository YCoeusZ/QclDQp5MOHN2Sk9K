from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline 
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score 
from xgboost import XGBClassifier 
import cloudpickle as cp
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from typing import Literal
import copy
import gzip

class model_eval: 
    def __init__(self, 
                #  pipe: Pipeline, 
                #  df_feat: pd.DataFrame, 
                #  df_tar: pd.DataFrame, 
                #  outer_cv: list, 
                #  param_dict: dict
                search_method: Literal["Random", "Grid"] = "Random", 
                search_cv_seed=420, 
                search_cv_splits=5, 
                search_seed=420, 
                search_cv_n_jobs=None
                ):
        # self.pipe=pipe
        # self.df_feat=df_feat
        # self.df_tar=df_tar
        # self.outer_cv=outer_cv
        # self.param_dict=param_dict
        self.search_method=search_method 
        self.search_seed=search_seed
        self.search_cv_seed=search_cv_seed
        self.search_cv=StratifiedKFold(n_splits=search_cv_splits, shuffle=True, random_state=self.search_cv_seed)
        self.search_cv_n_jobs=search_cv_n_jobs
        # self.fitted_dict_=None
        
    def _find_rfe_step_name(self, pipe: Pipeline): 
        for name, step in pipe.named_steps.items(): 
            if isinstance(step, RFE): 
                return name 
            else: 
                return None
    
    def eval(self, 
            df_feat, 
            df_tar, 
            pipe: Pipeline,
            outer_cv: list, 
            param_dict: dict
            # seed=420
            ): 
        self.fitted_dict_=dict()
        fold_num=0
        for tr_id, te_id in outer_cv: 
            param_dict_copy=copy.deepcopy(param_dict)
            fold_num=fold_num+1
            X_tr, X_te=df_feat.iloc[tr_id], df_feat.iloc[te_id]
            y_tr, y_te=np.ravel(df_tar.iloc[tr_id].values), np.ravel(df_tar.iloc[te_id].values)
            
            model_name=list(pipe.named_steps.keys())[-1]
            
            RFE_name=self._find_rfe_step_name(pipe)
            
            if model_name+"__scale_pos_weight" in param_dict_copy.keys(): 
                pos=(y_tr==1).sum()
                neg=(y_tr==0).sum()
                spw_raw=neg/max(pos,1) 
                spw_values=[0.7*spw_raw, spw_raw, 1.3*spw_raw]
                param_dict_copy[model_name+"__scale_pos_weight"]=spw_values
                if not RFE_name is None: 
                    param_dict_copy[RFE_name+"__estimator__scale_pos_weight"]=spw_values
            
            if self.search_method=="Random": 
                search=RandomizedSearchCV(
                    estimator=clone(pipe), 
                    param_distributions=param_dict_copy, 
                    n_iter=30, 
                    cv=self.search_cv, 
                    scoring=["f1", "roc_auc", "average_precision"], 
                    refit="average_precision", 
                    n_jobs=self.search_cv_n_jobs, 
                    verbose=1, 
                    random_state=self.search_seed, 
                    error_score="raise",
                    return_train_score=False
                )
                
            if self.search_method=="Grid": 
                search=GridSearchCV(
                    estimator=clone(pipe), 
                    param_grid=param_dict_copy, 
                    cv=self.search_cv, 
                    scoring=["roc_auc", "average_precision", "f1"], 
                    refit="average_precision", 
                    n_jobs=self.search_cv_n_jobs, 
                    verbose=1, 
                    error_score="raise",
                    return_train_score=False
                )
            
            search.fit(X=X_tr, y=y_tr) 
            best_pipe=search.best_estimator_
            # best_pred=best_pipe.predict(X=X_te)
            best_proba=best_pipe.predict_proba(X=X_te)[:,1]
            best_roc_auc=roc_auc_score(y_score=best_proba, y_true=y_te)
            best_ap=average_precision_score(y_score=best_proba, y_true=y_te)
            best_pred=(best_proba>=0.5).astype(int)
            best_f1=f1_score(y_pred=best_pred,y_true=y_te)
            self.fitted_dict_[f"Fold {fold_num}"]=(best_pipe, best_f1, best_roc_auc, best_ap, best_proba)
            # self.fitted_dict_explain_="The dictionary has key being the fold identifier, and a tuple value for each key. The indexes of the tuple, ordered by from 0 to 4, are:\n" +\
            #     "The best pipeline for this fold\n the f1 score of the best pipeline in this fold\n the roc auc of the best pipeline in this fold\n the average precision of the best pipeline in this fold\n " +\
            #     "the pred_proba produced by the pipeline in this fold. "
            
    def save_dict(self, 
                  save_path: str, compress: bool=False):
        if compress: 
            pickle_data=cp.dumps(self.fitted_dict_)
            print("Data pickled")
            compressed_data=gzip.compress(pickle_data)
            print("Data compressed")
            with open(save_path, "wb") as f: 
                f.write(compressed_data)
                print("Fitted dictionary saved") 
        else:  
            with open(save_path, "wb") as f: 
                cp.dump(self.fitted_dict_, f) 
                print("Fitted dictionary saved") 
            
    def load_dict(self, 
                  load_path: str, 
                  compress: bool=False,
                  force: bool=False): 
        if not force: 
            # len_cur_dict=len(self.fitted_dict_)
            if hasattr(self, "fitted_dict_"): 
                # print("WARNING: The fitted dictionary is NOT empty! \n")
                app_str="WARNING: The fitted dictionary is NOT empty! \n"
            else: 
                # print("The fitted dictionary is detected to be empty at this moment. \n")
                app_str="The fitted dictionary does NOT exist at this moment. \n"
            load_conf=input(f"{app_str} As a reminder: loading will over-ride existing fitted dictionary, please confirm to continue. (Y/N)").upper()
        if (load_conf == "Y") or (force): 
            if compress: 
                with open(load_path, "rb") as f: 
                    compressed_data=f.read()
                decompressed_data=gzip.decompress(compressed_data)
                self.fitted_dict_=cp.loads(decompressed_data)
                print("Loading completed. ")
            else: 
                with open(load_path, "rb") as f: 
                    self.fitted_dict_=cp.load(f)
                    # self.fitted_dict_explain_="The dictionary has key being the fold identifier, and a tuple value for each key. The indexes of the tuple, ordered by from 0 to 4, are:\n" +\
                    # "The best pipeline for this fold\n the f1 score of the best pipeline in this fold\n the roc auc of the best pipeline in this fold\n the average precision of the best pipeline in this fold\n " +\
                    # "the pred_proba produced by the pipeline in this fold. "
                print("Loading completed. ")
        else: 
            print("Loading aborted. ")
            
    def fitted_dict_explain(self): 
        print("The dictionary has key being the fold identifier, and a tuple value for each key. The indexes of the tuple, ordered by from 0 to 4, are:\n" +\
                " * The best pipeline for this fold\n * The f1 score of the best pipeline in this fold\n * The roc auc of the best pipeline in this fold\n * The average precision of the best pipeline in this fold\n " +\
                " * The pred_proba produced by the pipeline in this fold.\n")
        if not hasattr(self, "fitted_dict_"): 
            print("As a reminder, the attribute fitted_dict_ is NOT created at this moment. ")
            
    def pred_by_threshold(self, return_produced: bool=False): 
        self.pred_by_threshold_=dict()
        if hasattr(self, "fitted_dict_"): 
            for fold in self.fitted_dict_.keys(): 
                fold_proba=self.fitted_dict_[fold][4]
                threshold=np.arange(start=0, stop=1.01, step=0.01) 
                threshold=threshold[..., None] 
                pred_by_ths=fold_proba>=threshold
                self.pred_by_threshold_[fold]=pred_by_ths.astype(float)
                # self.pred_by_threshold_explain_="The dictionary has key being the fold identifier, and array value for each key.\n" +\
                # "The array value is, ordered from top to bottom, the prediction based on threshold >=0 to 1 with 0.01 step. " 
            if return_produced: 
                return self.pred_by_threshold_
        else: 
            if return_produced: 
                app_str="As a reminder, nothing is returned. "
            else: 
                app_str=""
            ValueError(f"Attribute fitted_dict_ does not exist, please train or load to create it. {app_str}")
    
    def pred_by_threshold_explain(self): 
        print("The dictionary has key being the fold identifier, and array value for each key.\n" +\
                "The array value is, ordered from top to bottom, the prediction based on threshold >=0 to 1 with 0.01 step.\n" )
        if not hasattr(self, "pred_by_threshold_"): 
            print("As a reminder, the attribute pred_by_threshold_ is NOT created at this moment. ")
            
    def confusion_data_by_threshold(self, y, outer_cv: list, return_produced: bool=False): 
        self.confusion_data_by_threshold_=dict()
        if hasattr(self, "fitted_dict_"): 
            if not hasattr(self, "pred_by_threshold_"): 
                self.pred_by_threshold(return_produced=False) 
            len_fitted_dict=len(self.fitted_dict_.keys())
            if len(outer_cv)!=len_fitted_dict: 
                ValueError("The outer_cv MUST be the same as the outer_cv used in the model_eval!") 
            if type(y)!= pd.DataFrame:
                y_c=pd.DataFrame({"y":y})
            else: 
                y_c=y.copy(deep=True)
            for fold in range(len_fitted_dict):
                fold_name=list(self.fitted_dict_.keys())[fold]
                fold_test_ind=outer_cv[fold][1] 
                fold_tar=np.ravel(y_c.iloc[fold_test_ind].values).astype(float)
                fold_pred=self.pred_by_threshold_[fold_name]
                fold_tar=np.repeat(fold_tar[None, ...], fold_pred.shape[0], axis=0)
                
                fold_pred_P=fold_pred==1
                fold_pred_N=fold_pred==0 
                fold_pred_T=fold_pred==fold_tar
                fold_pred_F=fold_pred!=fold_tar
                
                fold_TP=(fold_pred_T & fold_pred_P).sum(axis=1)
                fold_FP=(fold_pred_F & fold_pred_P).sum(axis=1)
                fold_FN=(fold_pred_F & fold_pred_N).sum(axis=1)
                fold_TN=(fold_pred_T & fold_pred_N).sum(axis=1)
                
                fold_dict={
                    "TP": fold_TP, 
                    "FP": fold_FP, 
                    "TN": fold_TN, 
                    "FN": fold_FN
                }
                
                self.confusion_data_by_threshold_[fold_name]=fold_dict
                # self.confusion_data_by_threshold_explain_="The dictionary has key being the fold identifier, and innder dictionary value for each key.\n"+\
                # "In the inner dictionary, for each key (identifying TP, FP, TN, and FN), values are the count of the corresponding category, ordered according to the threshold being from >=0 to >=1 with step 0.01."
        else: 
            if return_produced: 
                app_str="As a reminder, nothing is returned. " 
            else: 
                app_str=""
            ValueError(f"Attribute fitted_dict_ does not exist, please train or load to create it. {app_str}")
            
    def confusion_data_by_threshold_explain(self): 
        print("The dictionary has key being the fold identifier, and innder dictionary value for each key.\n"+\
                "In the inner dictionary, for each key (identifying TP, FP, TN, and FN), values are the count of the corresponding category, ordered according to the threshold being from >=0 to >=1 with step 0.01.\n")
        if not hasattr(self, "confusion_data_by_threshold_"): 
            print("As a reminder, the attribute confusion_data_by_threshold_ is NOT created at this moment. ")
            
    def metrics_by_threshold(self, eps: float=1e-10): 
        self.metrics_by_threshold_folds_=dict()
        # self.metrics_by_threshold_distribution_=dict()
        if not hasattr(self, "confusion_data_by_threshold_"): 
            ValueError("The confusion_data_by_threshold_ attribute does not exist at this moment. Please create the confusion data with object function confusion_data_by_threshold.")
        for fold in self.confusion_data_by_threshold_.keys(): 
            fold_confusion=self.confusion_data_by_threshold_[fold]
            
            fold_f1=(2*fold_confusion["TP"])/((2*fold_confusion["TP"])+fold_confusion["FP"]+fold_confusion["FN"]+eps)
            fold_precision=fold_confusion["TP"]/(fold_confusion["TP"]+fold_confusion["FP"]+eps)
            fold_FNR=fold_confusion["FN"]/(fold_confusion["FN"]+fold_confusion["TP"]+eps)
            
            self.metrics_by_threshold_folds_[fold]={
                "f1": fold_f1, 
                "precision": fold_precision, 
                "FNR": fold_FNR
            }
            
        f1_all=np.array([self.metrics_by_threshold_folds_[i]["f1"] for i in self.metrics_by_threshold_folds_.keys()])
        precision_all=np.array([self.metrics_by_threshold_folds_[i]["precision"] for i in self.metrics_by_threshold_folds_.keys()])
        FNR_all=np.array([self.metrics_by_threshold_folds_[i]["FNR"] for i in self.metrics_by_threshold_folds_.keys()])
            
        f1_mean=f1_all.mean(axis=0)
        f1_std=f1_all.std(axis=0)
        precision_mean=precision_all.mean(axis=0)
        precision_std=precision_all.std(axis=0)
        FNR_mean=FNR_all.mean(axis=0)
        FNR_std=FNR_all.std(axis=0)
        
        self.metrics_by_threshold_distribution_={
            "f1 mean": f1_mean, 
            "f1 std": f1_std, 
            "precision mean": precision_mean, 
            "precision std": precision_std, 
            "FNR mean": FNR_mean, 
            "FNR std": FNR_std
        }
        
        eval_f1=np.array([self.fitted_dict_[i][1] for i in self.fitted_dict_.keys()])
        eval_roc_auc=np.array([self.fitted_dict_[i][2] for i in self.fitted_dict_.keys()])
        eval_ap=np.array([self.fitted_dict_[i][3] for i in self.fitted_dict_.keys()])
        
        self.eval_metrics_={
            "f1": eval_f1.mean(), 
            "roc_auc": eval_roc_auc.mean(), 
            "ap": eval_ap.mean()
        }
    
    def create_visual(self): 
        if (not hasattr(self, "metrics_by_threshold_folds_")) or (not hasattr(self, "metrics_by_threshold_distribution_")): 
            self.metrics_by_threshold()
        #Create graph
        threshold=np.arange(start=0, stop=1.01, step=0.01) 
        fig, ax=plt.subplots(1,1, figsize=(12,6))
        sns.lineplot(x=threshold, y=self.metrics_by_threshold_distribution_["f1 mean"], color="black", label="f1 mean", ax=ax)
        sns.lineplot(x=threshold, y=self.metrics_by_threshold_distribution_["precision mean"], color="blue", label="precision mean", ax=ax)
        sns.lineplot(x=threshold, y=self.metrics_by_threshold_distribution_["FNR mean"], color="red", label="FNR mean", ax=ax)
        
        ax.set_title("Mean of key metrics by threshold")
        
        plt.show()