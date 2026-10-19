cd /home/dylan/AI/engine-lab/kernlab
for v in nm r1 r2; do echo "== $v"; ./kernlab mv_q4_$v.spv 16384 4096 $([ $v = r2 ] && echo 8 || echo 4) 8; done
echo "== real shape 10240x2560"
for v in nm r1 r2; do ./kernlab mv_q4_$v.spv 10240 2560 $([ $v = r2 ] && echo 8 || echo 4) 20; done
