#!/usr/bin/env bash
DATALIST=(cifar10)
BATCHSIZE=(100)
EPOCH=200
MODEL=resnet32
CHECKPOINTS=ckps
R=.pth.tar
DR=10
GPUS=(2 0 1 3)
ALGNAME=(SCCMA) # ACCSCCMA FastDRO PDSGD SCCMA dual_SGM PG_SMD2
LAMDA1=(10 5 7 9)
WLR=(0.2)
PLR=(1e-4 5e-5 5e-4)
IMRATIO=(0.01)
RHO=(1 0.5 0.01 0.05 )
BETA=(0.6)
CLASSTAU=(0.05 0.1 0.15)
AT=(0.5 0.8)
MYLAMBDA=(200 10)

for((wlr=0; wlr<1; wlr++)); do
{
for((agm=0; agm<1; agm++)); do
{
for((rpt=0; rpt<1; rpt++)); do
{
for((da=0; da<1; da++)); do
{
for((r=0; r<1; r++)); do
{
for((plr=0; plr<1; plr++)); do
{
for((lbd=0; lbd<4; lbd++)); do
{
for((ctu=0; ctu<1; ctu++)); do
{
for((at=0; at<1; at++));do
{
for((b=0; b<1; b++)); do
{


    python3 -W ignore main.py \
            --dataset ${DATALIST[$da]} \
            --model $MODEL \
            --saveFolder ${DATALIST[$da]}/${ALGNAME[$agm]}/Wm_${ALGNAME[$agm]}_wlr_${WLR[$agm]}_rho_${RHO[$r]}_beta_${BETA[$b]}_plr_${PLR[$plr]}_lambda1_${LAMDA1[$lbd]}_batch_${BATCHSIZE}_epochs_${EPOCH}_model_${MODEL}_DR_${DR}_Repeats_${rpt} \
            --res_filename ${DATALIST[$da]}_${ALGNAME[$agm]}_wlr_${WLR[$agm]}_rho_${RHO[$r]}_beta_${BETA[$b]}_plr_${PLR[$plr]}_lambda1_${LAMDA1[$lbd]}_batch_${BATCHSIZE}_epochs_${EPOCH}_model_${MODEL}_DR_${DR}_Repeats_${rpt}_class_tau_${CLASSTAU[$ctu]}  \
            --epochs ${EPOCH} \
            --batch-size ${BATCHSIZE} \
            --gpus ${GPUS[$lbd]}  \
            --lr ${WLR[$wlr]} \
            --restart_init_loop 1 \
            --lamda1 ${LAMDA1[$lbd]} \
            --epochs ${EPOCH} \
            --alg ${ALGNAME[$agm]} \
            --momentum 0.9 \
            --im_ratio ${IMRATIO[$da]} \
            --DR ${DR} \
            --sampleType uniform \
            --plr ${PLR[$plr]} \
            --rho ${RHO[$r]} \
            --beta ${BETA[$b]}\
            --class_tau ${CLASSTAU[$ctu]} \
            --lrlambda 0.00001 \
            --lamda0 1e-3 \
            --random_seed ${rpt} \
            --print_freq 50 \
            --mylambda ${MYLAMBDA[$at]}\
            --a_t ${AT[$at]} \
            --mvg_g_obj 1
}&
done
}
done
}
done
}
done
}
done
}
done
}
done
}
done
}
done
}
done
