import numpy as np

def inspect_npz(npz_path, preview_items=10):
    """
    查看 .npz 文件内容。

    参数:
        npz_path: str，npz 文件路径
        preview_items: int，每个数组预览前多少个元素
    """
    data = np.load(npz_path, allow_pickle=True)

    print(f"文件: {npz_path}")
    print(f"包含字段: {list(data.files)}")
    print("=" * 80)

    for key in data.files:
        arr = data[key]

        print(f"\n字段名: {key}")
        print(f"  type: {type(arr)}")
        print(f"  shape: {getattr(arr, 'shape', None)}")
        print(f"  dtype: {getattr(arr, 'dtype', None)}")

        # 预览数据
        try:
            flat = arr.reshape(-1)
            print(f"  preview: {flat[:preview_items]}")
        except Exception as e:
            print(f"  preview 失败: {e}")

        # 数值统计
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
            print(f"  min: {np.nanmin(arr)}")
            print(f"  max: {np.nanmax(arr)}")
            print(f"  mean: {np.nanmean(arr)}")
            print(f"  std: {np.nanstd(arr)}")

    data.close()


# 用法示例
inspect_npz("/Users/lily/Documents/myApps/Capstone_Saveme/15_aligner/data/ywj/npz/p2_2.npz")