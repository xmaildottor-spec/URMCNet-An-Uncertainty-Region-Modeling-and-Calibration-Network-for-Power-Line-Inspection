import os
import shutil

# ================= ⚙️ 配置区域 =================
# 1. 原始 JSON 文件夹的路径 (请将下面的路径替换为你自己的实际路径)
# 建议在路径字符串前加 'r'，防止 Windows 路径中的反斜杠 \ 引发转义错误
path_annotation_jsons = r'E:\code\dataset\TTPLA\ttpla\json'  

# 2. 划分列表文件 (.txt) 的绝对路径或相对路径
train_txt_path = r'E:\code\dataset\TTPLA\ttpla_dataset-master\splitting_dataset_txt\train.txt'
val_txt_path = r'E:\code\dataset\TTPLA\ttpla_dataset-master\splitting_dataset_txt\val.txt'
test_txt_path = r'E:\code\dataset\TTPLA\ttpla_dataset-master\splitting_dataset_txt\test.txt'

# 3. 输出文件夹的名称或路径 (默认在当前运行代码的目录下新建)
output_folder = r'E:\code\dataset\TTPLA\ttpla\splitting_jsons'
# ===============================================

# 定义输出子文件夹路径
train_folder = os.path.join(output_folder, 'train_jsons')
val_folder = os.path.join(output_folder, 'val_jsons')
test_folder = os.path.join(output_folder, 'test_jsons')

# 获取原始文件夹中所有的 json 文件名
jsons_names = [js for js in os.listdir(path_annotation_jsons) if js.endswith(".json")]

# 读取 txt 划分列表 (加入 encoding='utf-8' 防止乱码)
with open(train_txt_path, 'r', encoding='utf-8') as hndl: 
    train = [l.strip() for l in hndl]
with open(test_txt_path, 'r', encoding='utf-8') as hndl: 
    test = [l.strip() for l in hndl]
with open(val_txt_path, 'r', encoding='utf-8') as hndl: 
    val = [l.strip() for l in hndl]

# 打印统计信息，方便你检查是否读取正确
print(f"📝 列表统计: Train={len(train)}, Test={len(test)}, Val={len(val)}")
print(f"📂 原始文件夹中 JSON 总数: {len(jsons_names)}")
print(f"📊 TXT 列表中要求划分的总数: {len(train) + len(test) + len(val)}")

# 自动创建目标文件夹结构 (exist_ok=True 表示如果已存在也不会报错)
for p in [output_folder, train_folder, val_folder, test_folder]:
    os.makedirs(p, exist_ok=True)

print("🚀 开始复制文件...")

# 遍历原始 JSON 文件并安全复制到对应文件夹
for t in jsons_names:
    src_path = os.path.join(path_annotation_jsons, t)
    
    if t in train:
        shutil.copy(src_path, os.path.join(train_folder, t))
    elif t in test:
        shutil.copy(src_path, os.path.join(test_folder, t))
    elif t in val:
        shutil.copy(src_path, os.path.join(val_folder, t))

print("✅ 文件划分复制完成！")