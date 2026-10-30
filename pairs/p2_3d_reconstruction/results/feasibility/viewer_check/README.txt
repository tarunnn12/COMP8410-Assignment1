Viewer check of the two trained chair-video models (27 October 2026)

What was done: each trained model (nerfacto 2,500 iterations, splatfacto 15,000 iterations) was opened in the nerfstudio viewer with viewer_run.py
(the same loading fixes as the evaluation worker; GPU memory capped at 92 percent, job memory capped at 16 GB) and shown in the browser. The view was
rotated, zoomed and tilted with the mouse. The viewer's maximum resolution was left at its default of 512 pixels. Screenshots were taken after each
action and are saved here.

What was seen:
- splatfacto (splatfacto_viewer_1 to 5): after every drag, zoom and tilt the viewer showed 502 x 512 pixels; the chair stayed recognisable in every view close to it;
  in views that take in more of the room the edges of the room show streaks and floating fragments and the floor has orange smears.
- nerfacto (nerfacto_viewer_1 to 4): right after loading and right after a drag the viewer showed 29 x 30 pixels (blocky image); with the camera kept still
  the picture sharpened to 502 x 512 pixels in a time between 5 and 15 seconds (the screenshots were taken at those intervals).

What was not measured: a frame rate read from the viewer, and the response time of one drag. The evidence is the resolution that the viewer chose and the screenshots.
viewer_comparison.png puts three of the screenshots side by side.

More viewpoints (27 October 2026, later the same evening): further screenshots of both models were taken from other positions
(splatfacto_viewer_6 to 8, nerfacto_viewer_5 to 7; viewer_viewpoints.png puts all six side by side, top row splatfacto, bottom row nerfacto).
- Close to the chair both models show it clearly; away from it splatfacto breaks into stretched, spiky splats and nerfacto into white haze and dark
  patches near the corners of the room. Both leave bright floating patches and smears on the polished floor.
- The training cameras lie along one arc around the chair. The likely causes are that single arc, a background seen only at the edges of the
  frames and floor reflections; they were not tested one by one. The saved budget curve shows no gain in nerfacto PSNR from 2,500 to 10,000
  iterations and a plateau for splatfacto from 7,000 iterations.
