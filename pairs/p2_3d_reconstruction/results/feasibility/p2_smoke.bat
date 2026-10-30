@echo off
call "%~dp0env_nerf.bat"
set "OUT=C:\data\ns_out_smoke"
echo === splatfacto on poster (real capture), 300 iterations
%PYEXE% -c "import time;print('T0',time.time())"
%PYEXE% -m nerfstudio.scripts.train splatfacto --data C:\data\ns_sample\poster\transforms_subset.json --output-dir %OUT% --experiment-name poster_splat --timestamp smoke --max-num-iterations 300 --vis tensorboard --viewer.quit-on-train-completion True --steps-per-save 300 nerfstudio-data --downscale-factor 4
echo === exit %ERRORLEVEL%
echo === splatfacto on chair (synthetic), 300 iterations
%PYEXE% -m nerfstudio.scripts.train splatfacto --data C:\data\nerf_synth\chair --output-dir %OUT% --experiment-name chair_splat --timestamp smoke --max-num-iterations 300 --vis tensorboard --viewer.quit-on-train-completion True --steps-per-save 300 blender-data
echo === exit %ERRORLEVEL%
echo === nerfacto (torch) on chair, 100 iterations
%PYEXE% -m nerfstudio.scripts.train nerfacto --data C:\data\nerf_synth\chair --output-dir %OUT% --experiment-name chair_nerf --timestamp smoke --max-num-iterations 100 --vis tensorboard --viewer.quit-on-train-completion True --pipeline.model.implementation torch --steps-per-save 100 blender-data
echo === exit %ERRORLEVEL%
