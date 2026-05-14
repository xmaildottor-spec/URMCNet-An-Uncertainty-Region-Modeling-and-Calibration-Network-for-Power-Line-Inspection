import os
import cv2

# ================= ⚙️ 配置区域 =================
# 1. 输入的原始 Mask 基础文件夹路径 (包含训练集、验证集、测试集子文件夹)
input_base_dir = r'E:\code\dataset\TTPLA\ttpla\binary_gt_vis'

# 2. 调整尺寸后保存的新基础文件夹路径
output_base_dir = r'E:\code\dataset\TTPLA\ttpla\binary_gt_vis_512'

# 3. 目标尺寸 (宽, 高)
target_size = (512, 512)
# ===============================================

def resize_masks_recursive():
    print(f"🚀 开始递归调整 Mask 尺寸至 {target_size[0]}x{target_size[1]}...")
    
    if not os.path.exists(input_base_dir):
        print(f"❌ 错误：找不到输入文件夹 {input_base_dir}")
        return

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    count = 0

    # os.walk 会遍历 input_base_dir 下的每一个层级的文件夹
    for root, dirs, files in os.walk(input_base_dir):
        # 筛选出当前文件夹下的图片文件
        mask_files = [f for f in files if f.lower().endswith(valid_extensions)]
        
        if not mask_files:
            continue  # 如果当前子文件夹没有图片，直接跳过

        # 计算当前所在文件夹相对于 input_base_dir 的相对路径
        # 例如：如果是 '...\binary_gt\train'，相对路径就是 'train'
        rel_path = os.path.relpath(root, input_base_dir)
        
        # 在输出目录中拼接对应的结构：'...\binary_gt_512\train'
        target_dir = os.path.join(output_base_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for file_name in mask_files:
            input_path = os.path.join(root, file_name)
            
            # 以单通道灰度图模式读取 mask
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"⚠️ 无法读取图片，已跳过: {input_path}")
                continue

            # 核心：使用最近邻插值（INTER_NEAREST）进行缩放，防止产生 0 和 1 以外的杂波
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)

            # 强制保存为 .png 格式
            base_name = os.path.splitext(file_name)[0]
            output_png_path = os.path.join(target_dir, f"{base_name}.png")
            
            cv2.imwrite(output_png_path, resized_img)
            count += 1
            
            if count % 200 == 0:
                print(f"🔄 已处理 {count} 张图片...")

    print(f"\n🎉 尺寸调整全部完成！")
    print(f"✅ 成功处理: {count} 张图片")
    print(f"📂 新的数据集完全保持了原有的子文件夹结构，保存在: {output_base_dir}")

if __name__ == '__main__':
    resize_masks_recursive()