#!/usr/bin/env python3
"""
OmniGibson USD 文件解密工具
基于 OmniGibson 的实际解密方法
"""

import os
import sys
import tempfile
import shutil
from contextlib import contextmanager

from omnigibson.utils.asset_utils import decrypted


@contextmanager
def decrypt_usd_file(encrypted_path, output_path=None):
    """
    解密 USD 文件的上下文管理器
    基于 OmniGibson 的 decrypted 实现
    """
    if not os.path.exists(encrypted_path):
        raise FileNotFoundError(f"加密文件不存在: {encrypted_path}")
    
    # 使用 OmniGibson 的 decrypted 上下文管理器
    with decrypted(encrypted_path) as decrypted_path:
        if output_path and decrypted_path != output_path:
            # 如果需要复制到指定位置
            shutil.copy2(decrypted_path, output_path)
            yield output_path
        else:
            yield decrypted_path

def check_and_decrypt_usd(encrypted_path, output_dir=None):
    """
    检查并解密 USD 文件
    """
    print(f"🔍 检查加密文件: {encrypted_path}")
    
    # 检查文件是否存在
    if not os.path.exists(encrypted_path):
        print("❌ 错误: 文件不存在")
        return None
    
    # 检查文件格式
    try:
        with open(encrypted_path, 'rb') as f:
            first_bytes = f.read(100)
            if first_bytes.startswith(b'gAAAAAB'):
                print("✅ 确认是加密的 USD 文件")
            else:
                print("⚠️  文件可能不是加密格式，但尝试解密")
    except Exception as e:
        print(f"❌ 文件读取错误: {e}")
        return None
    
    # 设置输出路径
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(encrypted_path).replace('.encrypted.usd', '.usd')
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = None
    
    try:
        # 使用解密上下文管理器
        with decrypt_usd_file(encrypted_path, output_path) as decrypted_path:
            print(f"✅ 解密成功")
            print(f"📁 解密文件路径: {decrypted_path}")
            
            # 验证解密后的文件
            if verify_usd_file(decrypted_path):
                print("🎉 USD 文件验证通过！")
                return decrypted_path
            else:
                print("⚠️  解密文件验证失败")
                return None
                
    except Exception as e:
        print(f"❌ 解密失败: {e}")
        return None

def verify_usd_file(usd_path):
    """
    验证 USD 文件是否有效
    """
    try:
        from pxr import Usd
        
        print(f"🔧 验证 USD 文件: {usd_path}")
        
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print("❌ USD 文件打开失败")
            return False
        
        # 检查基本属性
        default_prim = stage.GetDefaultPrim()
        if default_prim:
            print(f"⭐ 默认 Prim: {default_prim.GetPath()}")
        else:
            print("⚠️  没有默认 Prim")
        
        # 计算 prim 数量
        prim_count = len(list(stage.Traverse()))
        print(f"🔢 Prim 数量: {prim_count}")
        
        if prim_count > 0:
            print("✅ USD 文件有效")
            return True
        else:
            print("⚠️  USD 文件中没有 prims")
            return False
            
    except Exception as e:
        print(f"❌ USD 验证错误: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python decrypt_usd.py <加密的usd文件> [输出目录]")
        print("示例: python decrypt_usd.py /path/to/file.encrypted.usd ./decrypted")
        sys.exit(1)
    
    encrypted_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 解密文件
    decrypted_path = check_and_decrypt_usd(encrypted_path, output_dir)
    
    if decrypted_path:
        print(f"\n🎉 解密完成！")
        print(f"解密文件: {decrypted_path}")
        
        # 提供使用示例
        print(f"\n💡 使用示例:")
        print(f"python3 test/usd2mjcf_test.py {decrypted_path} --generate_collision")
    else:
        print("\n💥 解密失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
