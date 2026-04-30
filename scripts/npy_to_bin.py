"""
将.npy文件转换为纯二进制文件，方便C++读取
"""

import numpy as np
import os

def npy_to_binary(npy_path, bin_path):
    """将.npy转换为纯二进制文件"""
    data = np.load(npy_path)
    print(f"  {npy_path}: dtype={data.dtype}, shape={data.shape}")
    
    if data.dtype == np.int64 or data.dtype == np.int32:
        data = data.astype(np.int32)
    else:
        data = data.astype(np.float32)
    
    data.tofile(bin_path)
    return data.shape, data.dtype

def convert_validation_data():
    """转换所有验证数据"""
    val_dirs = [
        "data/validation/single_token",
        "data/validation/short_text", 
        "data/validation/chinese_text",
        "data/validation/question",
        "data/validation/layer_activations"
    ]
    
    for val_dir in val_dirs:
        if not os.path.exists(val_dir):
            continue
            
        bin_dir = val_dir + "_bin"
        os.makedirs(bin_dir, exist_ok=True)
        
        for fname in os.listdir(val_dir):
            if fname.endswith('.npy'):
                npy_path = os.path.join(val_dir, fname)
                bin_path = os.path.join(bin_dir, fname.replace('.npy', '.bin'))
                shape, dtype = npy_to_binary(npy_path, bin_path)
                print(f"    -> {bin_path}")

if __name__ == "__main__":
    convert_validation_data()
    print("\n转换完成！")
