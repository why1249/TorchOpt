import unittest
import sys
import os

def run_all_tests():
    # 添加项目根目录到 Python 路径
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # 发现并运行所有测试
    test_loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    test_suite = test_loader.discover(start_dir, pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回退出码
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
