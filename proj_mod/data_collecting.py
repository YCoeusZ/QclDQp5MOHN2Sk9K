import gdown
import time 

def collect_data(id: str, store_path: str): 
    """
    A function to collect data from google drive. 
    
    :param id: The file id as a string. 
    :param store_path: Where you want to store the file. 
    :return: 0. 
    """
    start=time.perf_counter()
    url=f"https://drive.google.com/uc?id={id}"
    gdown.download(url=url, output=store_path, quiet=False)
    end=time.perf_counter()
    total=end-start
    print(f"Raw data downloaded in {total:.4f} seconds.")    
    return 0