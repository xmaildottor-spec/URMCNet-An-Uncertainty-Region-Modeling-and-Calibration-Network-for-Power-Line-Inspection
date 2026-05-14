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
    print(f"🚀 开始递归调整 Mask 尺寸至 {target_size[0]}x{target_size[1]} (开启防断裂模式)...")
    
    if not os.path.exists(input_base_dir):
        print(f"❌ 错误：找不到输入文件夹 {input_base_dir}")
        return

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    count = 0

    # os.walk 会遍历 input_base_dir 下的每一个层级的文件夹
    for root, dirs, files in os.walk(input_base_dir):
        mask_files = [f for f in files if f.lower().endswith(valid_extensions)]
        
        if not mask_files:
            continue

        rel_path = os.path.relpath(root, input_base_dir)
        target_dir = os.path.join(output_base_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for file_name in mask_files:
            input_path = os.path.join(root, file_name)
            
            # 以单通道灰度图模式读取 mask
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"⚠️ 无法读取图片，已跳过: {input_path}")
                continue

            # 🌟 核心修改区域：防止极细电力线断裂的降采样策略 🌟
            max_val = img.max() # 获取当前图像的最高像素值 (处理全黑背景图的异常)
            
            if max_val > 0:
                # 1. 使用 INTER_AREA 缩小，把细线晕染成灰色的过渡带，确保信息不丢失
                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                # 2. 极低阈值二值化：只要缩小后这个像素点有哪怕一丝丝灰度 (>0)，就强行设为前景值
                _, resized_img = cv2.threshold(resized_img, 0, max_val, cv2.THRESH_BINARY)
            else:
                # 如果是没有任何电力线的纯黑背景图，直接普通缩小即可
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