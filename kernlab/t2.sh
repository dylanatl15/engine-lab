cd /home/dylan/AI/engine-lab/kernlab
export KL_IQ4=1 KL_XPERM=1
for v in c1 c2 c1r2; do echo "== $v"; ./kernlab mv_iq4_$v.spv 16384 4096 $([ $v = c1r2 ] && echo 8 || echo 4) 8 | tail -2; ./kernlab mv_iq4_$v.spv 10240 2560 $([ $v = c1r2 ] && echo 8 || echo 4) 20 | tail -2; done
