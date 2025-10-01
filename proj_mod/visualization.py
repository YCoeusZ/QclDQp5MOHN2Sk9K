import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

def sns_kde(df: pd.DataFrame, feature: str) -> None: 
    sns.kdeplot(data=df[df["y"]==0], x=feature, fill=True, color="red", label="y == 0")
    sns.kdeplot(data=df[df["y"]==1], x=feature, fill=True, color="blue", label="y == 1")
    plt.xlabel(feature)
    plt.ylabel("KDE density")
    plt.legend()
    plt.show()
    
def barh_pct(df: pd.DataFrame, feature: str, figsize=(12,12)) -> None: 
    counts=df.groupby([feature,"y"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="barh", stacked=True, figsize=figsize, color={0:"red", 1:"blue"}, alpha=0.6)
    
    for i, (no_count, yes_count) in enumerate(zip(counts[0], counts[1])):
        total = no_count + yes_count
        pct_yes = (yes_count / total) * 100
        # pct_no = (no_count / total) * 100
        ax.text(x=no_count + yes_count + 0.2, y=i, s=f"{pct_yes:.2f}%", va="center", ha="center", color="blue", fontweight="bold", alpha=0.8)
        # ax.text(x=no_count/2, y=i, s=f"{pct_no:.2f}%", va="center", ha="center", color="white", fontweight="bold")
        
    plt.xlabel("Count")
    plt.ylabel(feature)
    plt.title("Yes/No counts with % Of each")
    plt.show()

def yes_pct(df: pd.DataFrame, feature: str, figsize=(6,12)) -> None: 
    yes_pct_df=df.groupby([feature])["y"].mean() * 100 
    plt.figure(figsize=figsize)
    sns.lineplot(x=yes_pct_df.values, y=yes_pct_df.index, color="blue", marker="o", orient="y") 
    plt.xlabel("Sub rate in percentage")
    plt.ylabel(feature)
    plt.grid(visible=True)
    plt.show()