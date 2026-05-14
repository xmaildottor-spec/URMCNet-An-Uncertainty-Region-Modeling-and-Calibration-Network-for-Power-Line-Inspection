import os
import json

# ================= ⚙️ 配置区域 =================
# 建议直接扫描原始的包含所有 JSON 的大文件夹，或者你划分好的任意一个子文件夹
# 这里以你的 train_jsons 为例
json_dir = r'E:\code\dataset\TTPLA\ttpla\splitting_jsons\train_jsons' 
# ===============================================

print("🔍 正在扫描 JSON 文件，提取所有唯一的类别标签...\n")

unique_labels = set()
file_count = 0

# 遍历文件夹内的所有文件
for file_name in os.listdir(json_dir):
    if file_name.endswith('.json'):
        file_path = os.path.join(json_dir, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取当前 JSON 中每一个 shape 的 label
            for shape in data.get('shapes', []):
                label = shape.get('label')
                if label:
                    # 使用 set 的 add 方法，会自动忽略重复的标签
                    unique_labels.add(label)
                    
            file_count += 1
        except Exception as e:
            print(f"⚠️ 读取 {file_name} 时出错: {e}")

print(f"✅ 扫描完毕！共检查了 {file_count} 个 JSON 文件。")
print(f"📊 发现该数据集中共有 {len(unique_labels)} 种不同的标签，分别是：\n")

# 为了方便查看，按字母排序后分行打印
for label in sorted(unique_labels):
    print(f"  - '{label}'")
    