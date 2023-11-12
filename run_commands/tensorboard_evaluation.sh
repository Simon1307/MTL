python tensorboard_eval/sigma_dist_dcuw_cuw_uw.py --dataset_name Cityscapes --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/cs_table2/ --num_iterations 111300 --num_runs_per_method 5 --num_tasks 2 --arch SegNet --result_path /home/kus1rng/MTL-playground/results/plots/sigma/cs_dcuw_cuw_uw_segnet_sigma.jpeg
python tensorboard_eval/sigma_dist_dcuw_cuw_uw.py --dataset_name CelebA --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/celebA_table2/ --num_iterations 63500 --num_runs_per_method 5 --num_tasks 40 --arch RESNET18 --result_path /home/kus1rng/MTL-playground/results/plots/sigma/celebA_dcuw_cuw_uw_resnet18_sigma.jpeg
python tensorboard_eval/sigma_dist_dcuw_cuw_uw.py --dataset_name NYU --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/nyuv2_table2/ --num_iterations 63600 --num_runs_per_method 5 --num_tasks 3 --arch RESNET50 --result_path /home/kus1rng/MTL-playground/results/plots/sigma/nyu_dcuw_cuw_uw_resnet50_sigma.jpeg

################## convergence plots
########## CS
### DSUW vs UW
python tensorboard_eval/convergence_plot.py --dataset_name Cityscapes --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/cs_table2/ --num_epochs 300 --num_runs_per_method 5 --arch SegNet --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/cs/cs_dsuw_uw_segnet_convergence.pdf --weighting_methods DCUW UW
python tensorboard_eval/convergence_plot.py --dataset_name Cityscapes --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/cs_table2/ --num_epochs 300 --num_runs_per_method 5 --arch RESNET50 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/cs/cs_dsuw_uw_resnet50_convergence.pdf --weighting_methods DCUW UW
python tensorboard_eval/convergence_plot.py --dataset_name Cityscapes --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/cityscapes_table2/ --num_epochs 300 --num_runs_per_method 5 --arch RESNET101 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/cs/cs_dsuw_uw_resnet101_convergence.pdf --weighting_methods DCUW UW
### DSUW vs SUW
python tensorboard_eval/convergence_plot.py --dataset_name Cityscapes --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/cs_table2/ --num_epochs 300 --num_runs_per_method 5 --arch SegNet --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/cs/cs_dsuw_suw_segnet_convergence.pdf --weighting_methods DCUW CUW
python tensorboard_eval/convergence_plot.py --dataset_name Cityscapes --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/cs_table2/ --num_epochs 300 --num_runs_per_method 5 --arch RESNET50 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/cs/cs_dsuw_suw_resnet50_convergence.pdf --weighting_methods DCUW CUW
python tensorboard_eval/convergence_plot.py --dataset_name Cityscapes --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/cityscapes_table2/ --num_epochs 300 --num_runs_per_method 5 --arch RESNET101 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/cs/cs_dsuw_suw_resnet101_convergence.pdf --weighting_methods DCUW CUW
########## NYU
### DSUW vs UW
python tensorboard_eval/convergence_plot.py --dataset_name NYU --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/nyuv2_table2/ --num_epochs 200 --num_runs_per_method 5 --arch SegNet --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/nyu/nyu_dsuw_uw_segnet_convergence.pdf --weighting_methods DCUW UW
python tensorboard_eval/convergence_plot.py --dataset_name NYU --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/nyuv2_table2/ --num_epochs 200 --num_runs_per_method 5 --arch RESNET50 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/nyu/nyu_dsuw_uw_resnet50_convergence.pdf --weighting_methods DCUW UW
python tensorboard_eval/convergence_plot.py --dataset_name NYU --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/nyu_table2/ --num_epochs 200 --num_runs_per_method 5 --arch RESNET101 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/nyu/nyu_dsuw_uw_resnet101_convergence.pdf --weighting_methods DCUW UW
### DSUW vs SUW
python tensorboard_eval/convergence_plot.py --dataset_name NYU --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/nyuv2_table2/ --num_epochs 200 --num_runs_per_method 5 --arch SegNet --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/nyu/nyu_dsuw_suw_segnet_convergence.pdf --weighting_methods DCUW CUW
python tensorboard_eval/convergence_plot.py --dataset_name NYU --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/nyuv2_table2/ --num_epochs 200 --num_runs_per_method 5 --arch RESNET50 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/nyu/nyu_dsuw_suw_resnet50_convergence.pdf --weighting_methods DCUW CUW
python tensorboard_eval/convergence_plot.py --dataset_name NYU --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/nyu_table2/ --num_epochs 200 --num_runs_per_method 5 --arch RESNET101 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/nyu/nyu_dsuw_suw_resnet101_convergence.pdf --weighting_methods DCUW CUW
########## CelebA
##### weight decay 1e-2
### DSUW vs UW
python tensorboard_eval/convergence_plot.py --dataset_name CelebA --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/celebA_table2_weight_decay/ --num_epochs 100 --num_runs_per_method 5 --arch RESNET18 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/celebA/celebA_dsuw_uw_resnet18_convergence_wd_1e-2.pdf --weighting_methods DCUW UW
### DSUW vs SUW
python tensorboard_eval/convergence_plot.py --dataset_name CelebA --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/celebA_table2_weight_decay/ --num_epochs 100 --num_runs_per_method 5 --arch RESNET18 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/celebA/celebA_dsuw_suw_resnet18_convergence_wd_1e-2.pdf --weighting_methods DCUW CUW
##### weight decay 1e-1
### DSUW vs UW
python tensorboard_eval/convergence_plot.py --dataset_name CelebA --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/celebA_table2_weight_decay_1e-1/ --num_epochs 100 --num_runs_per_method 5 --arch RESNET18 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/celebA/celebA_dsuw_uw_resnet18_convergence_wd_1e-1.pdf --weighting_methods DCUW UW
### DSUW vs SUW
python tensorboard_eval/convergence_plot.py --dataset_name CelebA --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/celebA_table2_weight_decay_1e-1/ --num_epochs 100 --num_runs_per_method 5 --arch RESNET18 --result_path /home/kus1rng/MTL-playground/final_results/plots/convergence/celebA/celebA_dsuw_suw_resnet18_convergence_wd_1e-1.pdf --weighting_methods DCUW CUW


python tensorboard_eval/delta_m_eval.py --dataset_name Cityscapes --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/cs_table2/ --num_runs_per_method 5 --num_tasks 2 --result_path /home/kus1rng/MTL-playground/results/tables/cs_loss_weighting_methods.tex
python tensorboard_eval/delta_m_eval.py --dataset_name CelebA --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/celebA_table2/ --num_runs_per_method 5 --num_tasks 40 --result_path /home/kus1rng/MTL-playground/results/tables/celebA_loss_weighting_methods.tex
python tensorboard_eval/delta_m_eval.py --dataset_name NYU --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/nyuv2_table2/ --num_runs_per_method 5 --num_tasks 3 --result_path /home/kus1rng/MTL-playground/results/tables/cnyuv2_loss_weighting_methods.tex
python tensorboard_eval/delta_m_eval.py --dataset_name Cityscapes --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/cityscapes_table3/ --num_runs_per_method 5 --num_tasks 2 --result_path /home/kus1rng/MTL-playground/results/tables/cs_complimentary_methods.tex
python tensorboard_eval/delta_m_eval.py --dataset_name NYU --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/nyu_table3/ --num_runs_per_method 5 --num_tasks 3 --result_path /home/kus1rng/MTL-playground/results/tables/nyuv2_complimentary_methods.tex
#python tensorboard_eval/delta_m_eval.py --dataset_name CelebA --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/celebA_table3/ --num_runs_per_method 5 --num_tasks 40 --result_path /home/kus1rng/MTL-playground/results/tables/celebA_complimentary_methods.tex



python tensorboard_eval/delta_m_eval.py --dataset_name CelebA --base_path /fs/scratch/rng_cr_bcai_dl_students/kus1rng/celebA_table2_weight_decay_dcuw_uw/ --num_runs_per_method 5 --num_tasks 40 --result_path /home/kus1rng/MTL-playground/results/celebA/celebA_table2_weight_decay_dcuw_uw.csv
