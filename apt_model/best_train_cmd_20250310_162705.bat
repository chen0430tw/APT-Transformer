@echo off
echo 使用Optuna优化后的最佳参数进行训练
echo 优化分数: 74.67/100
echo.
echo 首先修改配置文件...
copy /Y best_apt_config_20250310_162705.py "D:\apt_model\config\apt_config.py"
echo 配置文件已更新
echo.
cd D:\
echo 开始训练...
set APT_NO_STDOUT_ENCODING=1
python -m apt_model.main train ^
  --epochs 20 ^
  --batch-size 8 ^
  --learning-rate 2.418394206230888e-05 ^
  --save-path apt_model_best
echo.
echo 训练完成！
pause