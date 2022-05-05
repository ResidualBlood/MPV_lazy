### nlmeans降噪

import vapoursynth as vs
from vapoursynth import core

input = video_in

blur_mode = 2
frame_num = 1
radius_sw = 2
radius_snw = 2
strength = 3
GPU = 0

cut1 = core.fmtc.bitdepth(clip=input, bits=32)

if blur_mode == 1 :
	blur1 = core.rgvs.RemoveGrain(clip=cut1, mode=20)
	blur2 = core.rgvs.RemoveGrain(clip=blur1, mode=20)
	blur3 = core.rgvs.RemoveGrain(clip=blur2, mode=20)
if blur_mode == 2 :
	blur1 = core.std.Convolution(clip=cut1, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
	blur2 = core.std.Convolution(clip=blur1, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
	blur3 = core.std.Convolution(clip=blur2, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])

diff = core.std.MakeDiff(clipa=cut1, clipb=blur3)
cut2 = core.knlm.KNLMeansCL(clip=diff, d=frame_num, a=radius_sw, s=radius_snw, h=strength, wmode=2, device_type="GPU", device_id=GPU)
merge = core.std.MergeDiff(clipa=blur3, clipb=cut2)
output = core.fmtc.bitdepth(clip=merge, bits=16)

output.set_output()
