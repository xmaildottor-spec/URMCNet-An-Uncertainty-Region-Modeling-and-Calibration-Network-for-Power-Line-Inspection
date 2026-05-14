import os
import shutil

# ================= ⚙️ 配置区域 =================
# 1. 原始图片所在的文件夹路径
img_source_dir = r'E:\code\dataset\TTPLA\ttpla\img'

# 2. 之前已经划分好的 JSON 文件夹的基础路径
json_base_dir = r'E:\code\dataset\TTPLA\ttpla\splitting_jsons'

# 3. 新划分的图片准备存放的基础路径（这里建在 ttpla 目录下）
img_output_base = r'E:\code\dataset\TTPLA\ttpla\splitting_imgs'

# 4. 图片后缀名设定 (TTPLA 数据集默认通常是 .jpg)
# 如果你的图片是 .png，请将其修改为 '.png'
default_ext = '.jpg'
# ===============================================

# 自动推导子文件夹路径
splits = {
    'train': {
        'json_dir': os.path.join(json_base_dir, 'train_jsons'),
        'img_dir': os.path.join(img_output_base, 'train_imgs')
    },
    'val': {
        'json_dir': os.path.join(json_base_dir, 'val_jsons'),
        'img_dir': os.path.join(img_output_base, 'val_imgs')
    },
    'test': {
        'json_dir': os.path.join(json_base_dir, 'test_jsons'),
        'img_dir': os.path.join(img_output_base, 'test_imgs')
    }
}

print("🚀 开始匹配并复制图片文件...\n")

total_copied = 0

# 遍历划分 (train, val, test)
for split_name, paths in splits.items():
    json_dir = paths['json_dir']
    img_dir = paths['img_dir']
    
    # 检查 JSON 文件夹是否存在
    if not os.path.exists(json_dir):
        print(f"⚠️ 找不到 JSON 文件夹，已跳过: {json_dir}")
        continue
        
    # 创建对应的图片目标文件夹
    os.makedirs(img_dir, exist_ok=True)
    
    # 获取当前划分里所有的 json 文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    count = 0
    missing = 0
    
    for json_name in json_files:
        # 获取不带后缀的文件名 (例如: '001.json' -> '001')
        base_name = os.path.splitext(json_name)[0]
        
        # 拼接原始图片路径
        src_img_path = os.path.join(img_source_dir, base_name + default_ext)
        
        # 容错机制：如果默认后缀没找到，尝试找一下 .png 或 .jpeg
        if not os.path.exists(src_img_path):
            alt_exts = ['.png', '.jpeg', '.JPG', '.PNG']
            found = False
            for ext in alt_exts:
                alt_path = os.path.join(img_source_dir, base_name + ext)
                if os.path.exists(alt_path):
                    src_img_path = alt_path
                    default_ext = ext  # 更新后缀
                    found = True
                    break
            
            if not found:
                print(f"  ❌ 缺失: 找不到 JSON '{json_name}' 对应的图片文件")
                missing += 1
                continue
        
        # 目标图片路径
        dst_img_path = os.path.join(img_dir, os.path.basename(src_img_path))
        
        # 复制图片
        shutil.copy(src_img_path, dst_img_path)
        count += 1
        total_copied += 1
        
    print(f"✅ [{split_name}] 处理完毕: 成功复制 {count} 张图片. (缺失 {missing} 张)")

print(f"\n🎉 所有图片划分完成！共计复制图片 {total_copied} 张。")
print(f"📂 你可以在这里查看划分好的图片: {img_output_base}")