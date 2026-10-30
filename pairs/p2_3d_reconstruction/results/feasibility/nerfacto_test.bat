@echo off
call "%~dp0env_nerf.bat"
set "NS_OUT=C:\data\ns_out"
echo === ns-train nerfacto (torch implementation), 500 iterations
%PYEXE% -m nerfstudio.scripts.train nerfacto --data C:\data\ns_sample\poster\transforms_subset.json --output-dir %NS_OUT% --experiment-name poster_smoke --timestamp smoke --max-num-iterations 500 --vis tensorboard --viewer.quit-on-train-completion True --pipeline.model.implementation torch --steps-per-save 500 nerfstudio-data --downscale-factor 4
echo === exit code %ERRORLEVEL%
