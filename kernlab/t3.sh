cd /home/dylan/AI/engine-lab/kernlab
echo "-- old kernlab iq4 (dw fp16, flat x)"
KL_IQ4=1 KL_XPERM=1 ./kernlab mv_iq4_c1.spv 10240 2560 4 20 | tail -2 | head -1
KL_IQ4=1 KL_XPERM=1 ./kernlab mv_iq4_c1.spv 16384 4096 4 8 | tail -2 | head -1
echo "-- real layout"
KL_REAL=1 ./kernlab mv_real.spv 10240 2560 4 20 | tail -2 | head -1
KL_REAL=1 ./kernlab mv_real.spv 16384 4096 4 8 | tail -2 | head -1
