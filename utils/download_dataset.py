from datasets import load_dataset
import os

# 指定数据集名称
dataset_name = "BoKelvin/SLAKE"

# 保存路径
save_path = "checkpoints/datasets/slake"
os.makedirs(save_path, exist_ok=True)

print(f"开始下载数据集: {dataset_name}")
print("="*80)

try:
    # 下载数据集
    dataset = load_dataset(
        dataset_name,
        cache_dir="./cache",  # 可选：指定缓存目录
        trust_remote_code=True  # 如果数据集包含自定义代码
    )
    
    # 打印数据集信息
    print("\n数据集结构:")
    print(dataset)
    
    # 查看数据集各split的大小
    print("\n各split大小:")
    for split_name in dataset.keys():
        print(f"  - {split_name}: {len(dataset[split_name])} 条数据")
    
    # 查看第一条数据示例
    if len(dataset) > 0:
        first_split = list(dataset.keys())[0]
        print(f"\n{first_split} 的第一条数据示例:")
        print(dataset[first_split][0])
    
    # 保存数据集到磁盘
    print(f"\n正在保存到: {save_path}")
    dataset.save_to_disk(save_path)
    
    print("="*80)
    print(f"✓ 数据集已成功保存到 {save_path}")
    
except Exception as e:
    print(f"✗ 下载失败: {str(e)}")
    print("\n可能的解决方案:")
    print("1. 检查网络连接")
    print("2. 确认数据集名称正确")
    print("3. 检查是否需要 Hugging Face 登录: huggingface-cli login")
    print("4. 如果数据集是私有的,确保你有访问权限")