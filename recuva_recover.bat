@echo off
chcp 65001 >nul
echo ============================================================
echo Recuva - 恢复损坏的 test_*.py 文件
echo ============================================================
echo.

echo [1/3] 检查 Recuva 安装状态...
where recuva >nul 2>&1
if errorlevel 1 (
    echo    ❌ Recuva 未安装！
    echo.
    echo 请先下载并安装 Recuva:
    echo    下载: https://www.ccleaner.com/recuva/builds
    echo    或直接: https://download.ccleaner.com/rcvauc18.zip
    echo.
    pause
    exit /b 1
)

echo    ✅ Recuva 已安装
echo.

echo [2/3] 准备恢复...
set SOURCE=D:\APT-Transformer
set DEST=D:\APT-Transformer\recovered

echo    源目录: %SOURCE%
echo    输出目录: %DEST%
echo.

if not exist "%DEST%" mkdir "%DEST%"

echo 需要恢复的损坏文件列表:
echo.

echo   1. test_compile_backends.py
echo   2. test_compile_quick.py
echo   3. test_compile_small.py
echo   4. test_find_compiler.py
echo   5. test_gpu_final_v3.py
echo   6. test_gpu_simple.py
echo   7. test_gradient_flow.py
echo   8. test_int8_debug.py
echo   9. test_lecac.py
echo  10. test_lecac_int2_4_over_e.py
echo  11. test_lecac_int2_alpha_sweep.py
echo  12. test_lecac_int2_training.py
echo  13. test_lecac_int4.py
echo   14. test_lecac_int4_stats.py
echo   15. test_oom_comparison.py
echo   16. test_refactored_vb.py
echo   17. test_trace_gradient.py
echo   18. test_trace_ldbr.py
echo   19. test_triton_check.py
echo   20. test_va100_debug.py
echo   21. test_vb_nvlink_simulation.py
echo   22. test_vb_simple.py
echo   23. test_vb_training.py
echo   24. test_vb_training_speed_v6_2.py
echo   25. test_virtual_vram_v02.py
echo   26. test_vvram_backward.py
echo   27. test_vvram_debug.py
echo   28. test_vvram_peak_compare.py
echo.

echo [3/3] 启动 Recuva...
echo    恢复命令:
 recuva.exe "%SOURCE%\test_compile_backends.py" "%DEST%"
echo.
echo    注意: 如果文件已损坏，Recuva 会尝试深度扫描
echo.

pause

echo 正在启动 Recuva...
start /wait recuva.exe "%SOURCE%\test_compile_backends.py" "%DEST%"

echo.
echo ============================================================
echo 恢复完成！
echo 请检查 %DEST% 目录
echo.
echo 如果文件仍然损坏，尝试:
echo    1. 右键文件 → 属性 → 以前的版本
echo    2. 使用 git 历史恢复
echo ============================================================
pause
