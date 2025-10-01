import pandas as pd 
import datetime 
import numpy as np 
import heapq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif
from typing import Optional, Literal

def day_of_non_leap(month: str, day: int): 
    """
    Slow, consider using day_of_non_leap_np instead. 
    """
    date=datetime.datetime.strptime(f"{month}-{day}-2025","%b-%d-%Y") # 2025 is non leap 
    return date.timetuple().tm_yday 

def day_of_non_leap_np(month_arr:np.ndarray, day_arr:np.ndarray): 
    """
    Returns a np array of the day in a non leap year. 
    
    :return: The wanted array. 
    """
    month_start=np.array([0,31,59,90,120,151,181,212,243,273,304,334], dtype=int)
    return day_arr.astype(int) + month_start[(month_arr.astype(int)-1).astype(int)] 

def get_pct_group_by(df: pd.DataFrame, groupby: str) -> pd.DataFrame: 
    yes_pct_df=(df.groupby([groupby])["y"].mean() * 100).reset_index()
    yes_pct_df["count"]=df.groupby([groupby]).count().to_numpy()
    # dict_temp={key:value for key,value in zip(yes_pct_df[groupby].values, yes_pct_df["y"].values)}
    return yes_pct_df.sort_values(by=groupby, ascending=True)

def safe_return(key, in_dict: dict): 
    if key in list(in_dict.keys()): 
        return in_dict[key]
    else: 
        return np.mean(list(in_dict.values())) 
    
def closest_values(in_list: list, target, num_clst: int) -> list: 
    """
    Consider using mapp_fill_w_knn instead, this one is too slow. 
    """
    # # sorted_clst=sorted(in_list, key=(lambda x: abs(x-target)))
    # arr=np.array(in_list)
    # arr_abs_diff=np.abs(arr-target)
    # indices=np.argsort(arr_abs_diff)[:num_clst]
    # # return heapq.nsmallest(n=num_clst, iterable=in_list, key=(lambda x: abs(x-target)))
    # return arr[indices].tolist()
    radius=int(num_clst/2) 
    place=in_list.index(target) 
    if (place >= radius) and (place<=(len(in_list)-1-radius)): 
        out=in_list[place-radius:place+radius+1]
        out.remove(target)
    elif place < radius: 
        out=in_list[:num_clst+1]
        out.remove(target)
    else: 
        out=in_list[(len(in_list)-1-num_clst):]
        out.remove(target)
    return out

def safe_return_ord(key, in_dict: dict, num_clst: int = 2): 
    """
    Consider using map_fill_w_knn instead, this one is too slow.
    """
    #This is a bit slow 
    if key in list(in_dict.keys()): 
        return in_dict[key]
    else: 
        list_clst=closest_values(in_list=list(in_dict.keys()),target=key,num_clst=num_clst)
        list_value=[in_dict[value] for value in list_clst]
        return np.mean(list_value) 
    
def map_fill_w_knn(feature_arr: np.ndarray, cat_arr: np.ndarray, val_arr: np.ndarray, count_arr: np.ndarray, k: Optional[int] = None, cyc: bool = False, fall_back=None, present_override=False) -> np.ndarray: 
    """
    Creats a 1d numpy array with knn means (according to the correspondence suggested by that of cat_arr and val_arr) in order of that of feature_arr. If requested, this function can fill in with the present value if it is present in cat_arr, and fall_back if it is np.nan. 
    
    :param feature_arr: A numpy 1d arrary of the values classifying the feature. 
    :param cat_arr: The sorted 1d array of the all DISTINCT possible values for the feature. This array should NOT contain np.nan. 
    :param val_arr: The sorted (according to the corresponding order of cat_arr) 1d array of the values to be assigned. 
    :param count_arr: The sorted (according to the corresponding order of cat_arr) 1d array of the count of each category. 
    :param k: Defaulted to None, in which case it will be replaced with floor of sqrt of the length of cat_arr. The number of nearest neighbors one to calculate the average with. 
    :param cyc: Indicate if the values in cat_arr is cyclic, for instance, day in a non leap year is cyclic: one restarts with 1 after 365; but a person's age is not: Reincarnation is not yet scientifically proven. 
    :param fall_back: Defaulted to None. A value to fill the places in feature_arr with np.nan, one does not need this if one is confident that feature_arr does not have np.nan. 
    :return: desired numpy array
    """
    #Find where the value of feature is present in the recorded categories 
    anchor = np.searchsorted(cat_arr, feature_arr)
    leng = len(cat_arr)

    present = np.zeros_like(anchor, dtype=bool)
    in_range = anchor < leng
    present[in_range] = (cat_arr[anchor[in_range]] == feature_arr[in_range])
    #Set up the starting points of the neighborhoods: 
    if k is None: #Set default k 
        k=int(np.floor(np.sqrt(leng)))
    if k<1: 
        k=1
    if k>leng: 
        k=leng
    half=k//2
    if cyc: #cyc 
        starts=anchor-half #just start at the what it is, even if it is negative 
    else: #not cyc
        starts=np.clip(a=anchor-half, a_min=0, a_max=leng-k) #do not loop around
    
    #Find the neighborhood indices 
    n_offsets=np.arange(k) 
    if cyc: #cyc
        n_idx=(starts[:, None]+n_offsets[None, :]) % leng #if anything when beyond leng, loop it around. 
    else: #not cyc
        n_idx=(starts[:, None]+n_offsets[None, :]) 
    #Calculate the mean of knn 
    neigh_vals = val_arr[n_idx]
    neigh_wts = count_arr[n_idx]
    n_counts = neigh_wts.sum(axis=1)
    n_sums = (neigh_vals * neigh_wts).sum(axis=1)
    
    #Prepare for return 
    out=np.empty_like(n_sums, dtype=float)
    zero_mask=(n_counts == 0)
    out[~zero_mask] = n_sums[~zero_mask] / n_counts[~zero_mask]
    if zero_mask.any():
        out[zero_mask] = neigh_vals[zero_mask].mean(axis=1)
    
    #replace the places with np.nan with fall_back
    if np.issubdtype(feature_arr.dtype, np.number):
        nan_place = np.isnan(feature_arr)
        if nan_place.any():
            if fall_back is None:
                raise ValueError("feature_arr contains NaN, but fall_back is not provided.")
            out[nan_place] = fall_back
    #replace the value present with what they are already, if requested. 
    if present_override: 
        out[present]=val_arr[anchor[present]]
    
    return out 

def map_fill_fall_back(feature_arr: np.ndarray, cat_arr: np.ndarray, val_arr: np.ndarray, fall_back) -> np.ndarray: 
    """
    Creats a 1d numpy array with values assigned (according to the correspondence suggested by that of cat_arr and val_arr) in order of that of feature_arr. Fill in with the fall_back the value is unseen in cat_arr. 
    
    :param feature_arr: A numpy 1d arrary of the values classifying the feature. 
    :param cat_arr: The 1d array of the all distinct possible values for the feature. This array should NOT contain np.nan. 
    :param val_arr: The sorted (according to the corresponding order of cat_arr) 1d array of the values to be assigned. 
    :param fall_back: Defaulted to None. A value to fill the places in feature_arr with np.nan, one does not need this if one is confident that feature_arr does not have np.nan. 
    :return: desired numpy array
    """
    #Find where values are present 
    present=np.isin(element=feature_arr, test_elements=cat_arr)
    lookup={value: index for index, value in enumerate(cat_arr)}
    present_index=np.array([lookup.get(value, False) for value in feature_arr])
    
    #Create the array with fall_back values
    leng=len(feature_arr)
    out=np.full(shape=leng,fill_value=fall_back)
    
    #replace the places where the actual value is present 
    out[present]=val_arr[present_index[present]]
    return out 
    
    
class data_transform(BaseEstimator, TransformerMixin): 
    def __init__(self, per1: bool= True, cam1: bool = True, per2: bool = True, cam2: bool=True, cb: bool=True, cc: bool=True):
        # self.per1=personal_1
        # self.per2=personal_2
        # self.cam1=campaign_1
        # self.cam2=campaign_2
        # self.cb=clean_binary
        # self.cc=clean_cate
        self.per1=per1 
        self.per2=per2 
        self.cam1=cam1
        self.cam2=cam2 
        self.cb=cb 
        self.cc=cc
        
        # self.per=["age", "job", "marital", "education", "default", "balance", "housing", "loan"]
        # self.cam=["contact", "day", "month", "duration", "campaign"]
        pass 
    
    def fit(self, X: pd.DataFrame, y): 
        cate_drop_1=None
        if self.cc: 
            cate_drop_1="first"
        #Defensive deep copies 
        if type(y)!= pd.DataFrame:
            y_c=pd.DataFrame({"y":y})
        else: 
            y_c=y.copy(deep=True)
        X_c=X.copy(deep=True) 
        X_c=pd.concat([X_c.reset_index(drop=True), y_c.reset_index(drop=True)], axis=1)
        
        #fall back mean 
        self.fall_back_mean_=y_c["y"].mean()
        
        if self.per1 or self.per2: 
            self.per_binary_=["default", "housing", "loan"]
            self.per_cate_=["job", "marital"]
            self.per_ordcate_=["education"]
            self.per_numeric_=["age","balance"]
        
        if self.cam1 or self.cam2: 
            self.cam_cate_=["contact", "month"]
            self.cam_numeric_=["day", "day_of_year", "duration", "campaign"]
        
        #Personal data default encoding (1)
        if self.per1: 
            self.per_asis_=["age"]
            self.per_log_=["balance"]
            self.per_ord_=["education"]
            self.per_onehot_=["job","marital","default","housing","loan"] 
            #Nothing to learn for as is feature "age"
            #Nothing to learn for log feature "balance"
            #"Learn" the fixed mapping of ordinal feature "education" 
            self.per_ord_dict_=dict()
            self.per_ord_dict_["education"]=({"primary":0, "unknown": 1, "secondary": 2, "tertiary": 3},)
            #Learn the fitted one hot encoder of the one hot features 
            self.per_onehot_dict_=dict()
            for feature in self.per_onehot_: 
                ohencoder=OneHotEncoder(handle_unknown="ignore",drop=cate_drop_1) #We will use sparce_output=True as defaulted
                df_feature=X_c[[feature]]
                ohencoder.fit(X=df_feature)
                self.per_onehot_dict_[feature]=(ohencoder,)
        
        #Personal data target encoding (2) 
        if self.per2: 
            self.per_tar_mean_=["job", "marital", "education", "default", "housing", "loan"]
            self.per_tar_knn_=["age", "balance"] 
            #Learn the mapping dictionary for both fall back to knn mean and mean features 
            self.per_tar_mean_dict_=dict()
            for feature in self.per_tar_mean_: 
                self.per_tar_mean_dict_[feature]=(get_pct_group_by(df=X_c[[feature, "y"]],groupby=feature),)
            self.per_tar_knn_dict_=dict()
            for feature in self.per_tar_knn_: 
                self.per_tar_knn_dict_[feature]=(get_pct_group_by(df=X_c[[feature, "y"]],groupby=feature),)
            
        #Campaign data defaut encoding (1)    
        if self.cam1: 
            #Special step: Create "day_of_year" column
            if not "month_num" in X_c.columns: 
                X_c["month_num"]=X_c["month"].map({"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}).to_numpy().astype(int)
            if not "day_of_year" in X_c.columns: 
                X_c["day_of_year"]=day_of_non_leap_np(month_arr=X_c["month_num"].to_numpy(), day_arr=X_c["day"].to_numpy()).astype(int)
            self.cam_asis_=["day", "day_of_year", "campaign"] 
            self.cam_log_=["duration"]
            self.cam_onehot_=["contact", "month_num" ]
            #Nothing to learn for as is features 
            #Nothing to learn for log feature "duration" 
            #Learn the fitted one hot encoder of the one hot features 
            self.cam_onehot_dict_=dict()
            for feature in self.cam_onehot_: 
                ohencoder=OneHotEncoder(handle_unknown="ignore", drop=cate_drop_1) 
                df_feature=X_c[[feature]]
                ohencoder.fit(X=df_feature)
                self.cam_onehot_dict_[feature]=(ohencoder,)
                
        #Campaign data target encoding (2)
        if self.cam2: 
            #Create the "day_of_year" feature if target encoding on campaign features are required (we do not need it for the default as is encoding, as there is nothing to learn). 
            # X_c["day_of_year"]=X_c[["month","day"]].apply(lambda row: day_of_non_leap(month=row["month"],day=row["day"]),axis=1)
            if not "month_num" in X_c.columns: 
                X_c["month_num"]=X_c["month"].map({"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}).to_numpy().astype(int)
            if not "day_of_year" in X_c.columns: 
                X_c["day_of_year"]=day_of_non_leap_np(month_arr=X_c["month_num"].to_numpy(), day_arr=X_c["day"].to_numpy()).astype(int)
            self.cam_tar_mean_=["contact"]
            self.cam_tar_knn_cyc_=["day", "month_num", "day_of_year"] 
            self.cam_tar_knn_ncyc_=["duration", "campaign"] 
            #Learn mapping dictionary for all categories of features 
            self.cam_tar_mean_dict_=dict()
            for feature in self.cam_tar_mean_: 
                self.cam_tar_mean_dict_[feature]=(get_pct_group_by(df=X_c[[feature, "y"]], groupby=feature),)
            self.cam_tar_knn_cyc_dict_=dict()
            for feature in self.cam_tar_knn_cyc_: 
                self.cam_tar_knn_cyc_dict_[feature]=(get_pct_group_by(df=X_c[[feature, "y"]], groupby=feature),)
            self.cam_tar_knn_ncyc_dict_=dict()
            for feature in self.cam_tar_knn_ncyc_: 
                self.cam_tar_knn_ncyc_dict_[feature]=(get_pct_group_by(df=X_c[[feature, "y"]],groupby=feature),)
        
        return self
                
    def transform(self, X: pd.DataFrame): # -> pd.DataFrame: 
        
        #Defensive deep copy 
        X_c=X.copy(deep=True)
        out=pd.DataFrame()
        #Personal data default encoding (1)
        if self.per1: 
            #As is
            for feature in self.per_asis_: 
                out[feature]=X_c[feature].values
            # print(out.shape)
            #Log
            for feature in self.per_log_: 
                eps=0.1
                out[feature]=np.sign(X_c[feature].to_numpy()+1)*np.log(np.abs(X_c[feature].to_numpy()+1)+eps)
            # print(out.shape)
            #Ordinal 
            for feature in self.per_ord_: 
                out[feature]=X_c[feature].map(self.per_ord_dict_[feature][0]).values
            # print(out.shape)
            #Onehot
            for feature in self.per_onehot_: 
                ohencoder=self.per_onehot_dict_[feature][0]
                encoded_sparse=ohencoder.transform(X_c[[feature]])
                encoded_df=pd.DataFrame(
                    encoded_sparse.toarray(), 
                    columns=ohencoder.get_feature_names_out([feature])
                )
                # print("one_hot:", encoded_df.shape)
                out=pd.concat([out.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
                # print(out.shape)
            # print(out.shape)
                
        if self.per2: 
            for feature in self.per_tar_mean_: 
                tar=self.per_tar_mean_dict_[feature][0]
                out[feature+"_tar"]=map_fill_fall_back(feature_arr=X_c[feature].to_numpy(), cat_arr=tar[feature].to_numpy(), val_arr=tar["y"].to_numpy(), fall_back=self.fall_back_mean_)
            for feature in self.per_tar_knn_: 
                tar=self.per_tar_knn_dict_[feature][0]
                out[feature+"_tar"]=map_fill_w_knn(feature_arr=X_c[feature].to_numpy(), cat_arr=tar[feature].to_numpy(), val_arr=tar["y"].to_numpy(), count_arr=np.log(tar["count"].to_numpy()+1))
                
        if self.cam1: 
            #Special step: Create "day_of_year" column and "month_num"
            if not "month_num" in X_c.columns: 
                X_c["month_num"]=X_c["month"].map({"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}).astype(int).values
            if not "day_of_year" in X_c.columns: 
                X_c["day_of_year"]=day_of_non_leap_np(month_arr=X_c["month_num"].to_numpy(), day_arr=X_c["day"].to_numpy()).astype(int)
            for feature in self.cam_asis_: 
                out[feature]=X_c[feature].values
            for feature in self.cam_log_: 
                out[feature]=np.log(X_c[feature].to_numpy()+1)
            for feature in self.cam_onehot_: 
                ohencoder=self.cam_onehot_dict_[feature][0]
                encoded_sparse=ohencoder.transform(X_c[[feature]])
                encoded_df=pd.DataFrame(
                    encoded_sparse.toarray(), 
                    columns=ohencoder.get_feature_names_out([feature])
                )
                out=pd.concat([out.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
            
        if self.cam2: 
            #Special step: Create "day_of_year" column and "month_num"
            if not "month_num" in X_c.columns: 
                X_c["month_num"]=X_c["month"].map({"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}).astype(int).values
            if not "day_of_year" in X_c.columns: 
                X_c["day_of_year"]=day_of_non_leap_np(month_arr=X_c["month_num"].to_numpy(), day_arr=X_c["day"].to_numpy()).astype(int)
            for feature in self.cam_tar_mean_: 
                tar=self.cam_tar_mean_dict_[feature][0]
                out[feature+"_tar"]=map_fill_fall_back(feature_arr=X_c[feature].to_numpy(), cat_arr=tar[feature].to_numpy(), val_arr=tar["y"].to_numpy(), fall_back=self.fall_back_mean_)
            for feature in self.cam_tar_knn_cyc_: 
                tar=self.cam_tar_knn_cyc_dict_[feature][0]
                out[feature+"_tar"]=map_fill_w_knn(feature_arr=X_c[feature].to_numpy(), cat_arr=tar[feature].to_numpy(), val_arr=tar["y"].to_numpy(), count_arr=np.log(tar["count"].to_numpy()+1), cyc=True)
            for feature in self.cam_tar_knn_ncyc_: 
                tar=self.cam_tar_knn_ncyc_dict_[feature][0]
                out[feature+"_tar"]=map_fill_w_knn(feature_arr=X_c[feature].to_numpy(),cat_arr=tar[feature].to_numpy(), val_arr=tar["y"].to_numpy(), count_arr=np.log(tar["count"].to_numpy()+1))
                
        if self.cb and self.per1 and self.per2: 
            lst_drop=[]
            all_feat=out.columns
            for feature in self.per_binary_: 
                lst_sub_feat=[sub_feat for sub_feat in all_feat if feature in sub_feat]
                lst_drop+=lst_sub_feat[1:]
            out.drop(labels=lst_drop, axis=1, inplace=True)
        
        return out 
    
class data_selector(BaseEstimator, TransformerMixin): 
    def __init__(self, cut: float, how: Literal["f score", "mi score"], mi_n_jobs=None, fav_tar: bool=False, for_tar: bool=False, fr: list=[]):
        # self.how=how 
        # self.fav_tar=favor_target
        # self.for_tar=force_target
        # self.cut=colinearity_threshold
        # self.fr=force_remove
        # self.mi_n_jobs=mi_n_jobs
        self.how=how 
        self.fav_tar=fav_tar
        self.for_tar=for_tar
        self.cut=cut
        self.fr=fr
        self.mi_n_jobs=mi_n_jobs
        
    def fit(self, X, y): 
        if type(y)!= pd.DataFrame:
            y_c=pd.DataFrame({"y":y})
        else: 
            y_c=y.copy(deep=True)
        X_c=X.copy(deep=True) 
        if self.how == "f score": 
            scaler= StandardScaler() 
            f_score, p_value= f_classif(X=scaler.fit_transform(X=X_c),y=y_c["y"])
            df_f=pd.DataFrame({
                "features": X_c.columns, 
                "f score": f_score.round(2), 
                "p value": p_value.round(2)
            })
            df_f["rank"] = df_f["f score"].rank(method="dense", ascending=False).astype(int)
            self.df_score_ranked_=df_f.sort_values(by="rank",ascending=True).reset_index(drop=True)
        elif self.how == "mi score": 
            scaler= StandardScaler()
            mi_score=mutual_info_classif(X=scaler.fit_transform(X=X_c),y=y_c["y"], n_neighbors=10, random_state=420, n_jobs=self.mi_n_jobs)
            df_mi=pd.DataFrame({
                "features": X_c.columns, 
                "MI": mi_score
            })
            df_mi["rank"]=df_mi["MI"].rank(method="dense",ascending=False).astype(int)
            self.df_score_ranked_=df_mi.sort_values(by="rank", ascending=True).reset_index(drop=True)
        features_ranked=self.df_score_ranked_["features"].to_list()
        
        X_c_ordered=X_c[features_ranked]
        df_feat_corr=X_c_ordered.corr()
        df_feat_corr_cut=df_feat_corr[df_feat_corr.abs()>=self.cut]
        
        set_remove=set(self.fr)
        
        for feature in features_ranked: 
            if (feature in set_remove): 
                if (not self.for_tar): 
                    continue
                elif (self.for_tar) and ("_tar" not in feature): 
                    continue
            cur_df_feat_corr=df_feat_corr_cut[[feature]]
            cur_feat_corr_list=list(cur_df_feat_corr.dropna().index)
            cur_feat_index=cur_feat_corr_list.index(feature)
            cur_feat_tar=("_".join(feature.split("_")[:-1]))+"_tar"
            following_cur_feat=cur_feat_corr_list[cur_feat_index:]
            if len(following_cur_feat)==1: 
                continue
            elif (not self.fav_tar) or ("_tar" in feature and self.fav_tar): 
                set_remove.update(cur_feat_corr_list[cur_feat_index+1:])
            elif cur_feat_tar in following_cur_feat: 
                following_cur_feat.remove(cur_feat_tar)
                set_remove.update(following_cur_feat)
            else: 
                set_remove.update(cur_feat_corr_list[cur_feat_index+1:])
                
        self.feat_keep_=[x for x in features_ranked if x not in set_remove]

        if self.for_tar: 
            list_tar=[x for x in features_ranked if "_tar" in x]
            self.feat_keep_=list(set(self.feat_keep_+list_tar))
        
        return self
            
    def transform(self, X): 
        X_c=X.copy(deep=True) 
        return X_c[self.feat_keep_]