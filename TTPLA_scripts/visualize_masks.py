import os
import cv2
import numpy as np

# ================= ⚙️ 配置区域 =================
mask_base_dir = r'E:\code\dataset\TTPLA\ttpla\splitting_masks'
vis_output_base = r'E:\code\dataset\TTPLA\ttpla\splitting_vis_masks'

# 3. 颜色映射表 (OpenCV 默认使用的是 BGR 格式！)
COLOR_MAP = {
    0: (0, 0, 0),       # 背景: 黑色
    1: (150, 51, 255),     # cable (电力线): 红色 (B=0, G=0, R=255)
    2: (0, 255, 0),     # tower_lattice (桁架杆塔): 绿色 (B=0, G=255, R=0)
    3: (255, 0, 0),     # tower_tucohy (管型杆塔): 蓝色 (B=255, G=0, R=0)
    4: (0, 255, 255)    # tower_wooden (木质电线杆): 黄色 (B=0, G=255, R=255)
}
# ===============================================

splits = {
    'train': {
        'in_dir': os.path.join(mask_base_dir, 'train_masks'),
        'out_dir': os.path.join(vis_output_base, 'train_vis_masks')
    },
    'val': {
        'in_dir': os.path.join(mask_base_dir, 'val_masks'),
        'out_dir': os.path.join(vis_output_base, 'val_vis_masks')
    },
    'test': {
        'in_dir': os.path.join(mask_base_dir, 'test_masks'),
        'out_dir': os.path.join(vis_output_base, 'test_vis_masks')
    }
}

print("🚀 开始为细分类别的 Mask 标签图上色并生成可视化图片...\n")

total_vis = 0

for split_name, paths in splits.items():
    in_dir = paths['in_dir']
    out_dir = paths['out_dir']
    
    if not os.path.exists(in_dir):
        print(f"⚠️ 找不到输入文件夹，已跳过: {in_dir}")
        continue
        
    os.makedirs(out_dir, exist_ok=True)
    
    mask_files = [f for f in os.listdir(in_dir) if f.endswith('.png')]
    count = 0
    
    for mask_name in mask_files:
        in_path = os.path.join(in_dir, mask_name)
        out_path = os.path.join(out_dir, mask_name)
        
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"❌ 无法读取图片: {mask_name}")
            continue
            
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 批量颜色替换
        for class_id, color in COLOR_MAP.items():
            if class_id == 0:
                continue
            color_mask[mask == class_id] = color
            
        cv2.imwrite(out_path, color_mask)
        count += 1
        total_vis += 1
        
    print(f"✅ [{split_name}] 处理完毕: 成功生成 {count} 张彩色可视化图片。")

print(f"\n🎉 所有图片上色完成！共计生成彩色图片 {total_vis} 张。")
print(f"📂 你的可视化图片存放在这里: {vis_output_base}")