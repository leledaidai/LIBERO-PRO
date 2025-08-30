#!/usr/bin/env python3
"""
USD 文件检查工具
用于验证 USD 文件是否正确并可被 USD 库读取
"""

import os
import sys
import argparse
from pxr import Usd, Tf

def check_usd_file(usd_path):
    """
    检查 USD 文件的完整性和可读性
    """
    print(f"检查 USD 文件: {usd_path}")
    print("=" * 60)
    
    # 1. 检查文件是否存在
    if not os.path.exists(usd_path):
        print("❌ 错误: 文件不存在")
        return False
    
    # 2. 检查文件权限
    if not os.access(usd_path, os.R_OK):
        print("❌ 错误: 文件不可读，请检查权限")
        return False
    
    # 3. 检查文件大小
    file_size = os.path.getsize(usd_path)
    print(f"📄 文件大小: {file_size} 字节")
    
    if file_size == 0:
        print("❌ 错误: 文件为空")
        return False
    
    # 4. 尝试打开 USD 文件
    try:
        print("🔄 尝试打开 USD 文件...")
        stage = Usd.Stage.Open(usd_path)
        
        if not stage:
            print("❌ 错误: Usd.Stage.Open 返回 None")
            return False
        
        # 5. 检查 stage 的基本属性
        print("✅ USD 文件成功打开")
        print(f"📋 根图层: {stage.GetRootLayer().identifier}")
        print(f"🎯 起始时间码: {stage.GetStartTimeCode()}")
        print(f"⏰ 结束时间码: {stage.GetEndTimeCode()}")
        
        # 6. 检查 prims 数量
        prim_count = len(list(stage.Traverse()))
        print(f"🔢 Prim 数量: {prim_count}")
        
        if prim_count == 0:
            print("⚠️  警告: 文件中没有找到任何 prim")
        
        # 7. 检查默认 prim
        default_prim = stage.GetDefaultPrim()
        if default_prim:
            print(f"⭐ 默认 Prim: {default_prim.GetPath()}")
        else:
            print("⚠️  警告: 没有设置默认 prim")
        
        # 8. 检查文件格式
        file_format = Usd.UsdFileFormat.FindByExtension(usd_path)
        if file_format:
            print(f"📁 文件格式: {file_format.formatId}")
        
        # 9. 尝试读取一些 prims
        print("\n🔍 扫描前10个 prims:")
        for i, prim in enumerate(stage.Traverse()):
            if i >= 10:  # 只显示前10个
                break
            prim_type = prim.GetTypeName() or "未定义"
            print(f"  {i+1}. {prim.GetPath()} ({prim_type})")
        
        if prim_count > 10:
            print(f"  ... 还有 {prim_count - 10} 个 prims")
        
        return True
        
    except Tf.ErrorException as e:
        print(f"❌ USD 库错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

def check_usd_environment():
    """
    检查 USD 环境配置
    """
    print("🔧 检查 USD 环境配置")
    print("-" * 40)
    
    try:
        # 检查 USD 版本
        usd_version = Usd.GetVersion()
        print(f"✅ USD 版本: {usd_version[0]}.{usd_version[1]}.{usd_version[2]}")
        
        # 尝试创建简单的 USD 文件来测试环境
        test_stage = Usd.Stage.CreateInMemory()
        sphere = Usd.Sphere.Define(test_stage, "/TestSphere")
        if sphere:
            print("✅ USD 环境正常")
            return True
        else:
            print("❌ USD 环境异常")
            return False
            
    except Exception as e:
        print(f"❌ USD 环境检查失败: {e}")
        return False

def analyze_file_content(usd_path):
    """
    分析文件内容（如果是文本格式）
    """
    print("\n📊 文件内容分析:")
    print("-" * 40)
    
    try:
        with open(usd_path, 'rb') as f:
            first_bytes = f.read(1024)
        
        # 检查文件签名
        if first_bytes.startswith(b'PXR-USDC'):
            print("✅ 二进制 USD 文件 (USDC)")
        elif first_bytes.startswith(b'#usda'):
            print("✅ 文本 USD 文件 (USDA)")
        else:
            # 尝试解码为文本
            try:
                first_lines = first_bytes.decode('utf-8').split('\n')[:5]
                if any('#usda' in line for line in first_lines):
                    print("✅ 文本 USD 文件 (USDA)")
                else:
                    print("⚠️  未知文件格式，可能不是标准的 USD 文件")
                    print("前几行内容:")
                    for i, line in enumerate(first_lines[:3]):
                        print(f"  {i+1}: {line.strip()}")
            except UnicodeDecodeError:
                print("🔒 二进制文件，无法读取文本内容")
                
    except Exception as e:
        print(f"❌ 文件内容分析失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='USD 文件检查工具')
    parser.add_argument('usd_file', help='要检查的 USD 文件路径')
    parser.add_argument('--check-env', action='store_true', 
                       help='同时检查 USD 环境')
    args = parser.parse_args()
    
    usd_path = os.path.expanduser(args.usd_file)
    
    if not os.path.exists(usd_path):
        print(f"错误: 文件 '{usd_path}' 不存在")
        sys.exit(1)
    
    # 检查环境（可选）
    if args.check_env:
        env_ok = check_usd_environment()
        if not env_ok:
            print("USD 环境有问题，请先修复环境")
            sys.exit(1)
    
    # 分析文件内容
    analyze_file_content(usd_path)
    
    # 检查 USD 文件
    success = check_usd_file(usd_path)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 USD 文件检查通过！")
        sys.exit(0)
    else:
        print("💥 USD 文件检查失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
